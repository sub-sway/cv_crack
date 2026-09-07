from flask import Flask, Response, jsonify, render_template
from threading import Thread, Lock

import depthai as dai
import numpy as np
import cv2
import json
import time
import torch
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# 설정
# ============================================================
MODEL_PATH = "runs/segment/runs/dacl10k/yolo11n_seg_1024-2/weights/best.pt"

WIDTH = 640
HEIGHT = 480
FPS = 30

CONF_THRESHOLD = 0.35
YOLO_IMGSZ = 640
WALL_KERNEL_SIZE = 15
TORCH_THREADS = 6

# Depth 신뢰도 판정 기준
MIN_DEPTH_VALID_RATIO = 0.30
MIN_RECESSED_PIXEL_RATIO = 0.10
MIN_DEPTH_SIGNAL_MM = 5.0
DEPTH_NOISE_MULTIPLIER = 3.0
DEPTH_PERCENTILE = 90
LOCAL_DEPTH_VIS_MAX_MM = 30.0

# 치수 보정 및 보수비 설정
ARUCO_MARKER_SIZE_MM = 50.0
ARUCO_DETECT_INTERVAL = 5
REPAIR_COST_CONFIG_PATH = Path("repair_cost_config.json")
CAPTURE_DIR = Path("captures")

# 위험 기준 (교량 점검 기준서 기반)
CRITICAL_WIDTH_MM = 0.3 

# ============================================================
# 헬퍼 함수
# ============================================================
def deproject_pixels_to_points_mm(xs, ys, depths_mm, fx, fy, cx, cy):
    """여러 Depth 픽셀을 한 번의 NumPy 연산으로 3D 좌표로 변환합니다."""
    z = depths_mm.astype(np.float32, copy=False)
    x = (xs.astype(np.float32, copy=False) - cx) * z / fx
    y = (ys.astype(np.float32, copy=False) - cy) * z / fy
    return np.column_stack((x, y, z))


def load_repair_cost_config():
    if not REPAIR_COST_CONFIG_PATH.exists():
        return {"currency": "KRW", "methods": {}}

    with REPAIR_COST_CONFIG_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def detect_aruco_scale(color_image, detector):
    """동일 평면의 50mm ArUco 마커로 평균 mm/픽셀 비율을 구합니다."""
    corners, ids, _ = detector.detectMarkers(color_image)
    if ids is None or len(corners) == 0:
        return None, None, None

    # 여러 마커가 있으면 픽셀 면적이 가장 큰(가장 선명한) 마커 사용
    marker_areas = [abs(cv2.contourArea(corner.reshape(4, 2))) for corner in corners]
    marker_index = int(np.argmax(marker_areas))
    marker_corners = corners[marker_index].reshape(4, 2)
    side_lengths = [
        np.linalg.norm(marker_corners[(index + 1) % 4] - marker_corners[index])
        for index in range(4)
    ]
    marker_pixel_size = float(np.mean(side_lengths))
    if marker_pixel_size <= 0:
        return None, None, None

    mm_per_pixel = ARUCO_MARKER_SIZE_MM / marker_pixel_size
    marker_id = int(np.asarray(ids).reshape(-1)[marker_index])
    return mm_per_pixel, marker_corners, marker_id


def select_repair_method(crack_width_mm):
    """영상 폭 기준으로 임시 보수 공법을 선택합니다."""
    if crack_width_mm >= CRITICAL_WIDTH_MM:
        return "injection"
    return "surface_treatment"


def estimate_repair_cost(length_mm, width_mm, cost_config):
    method_key = select_repair_method(width_mm)
    method_config = cost_config.get("methods", {}).get(method_key, {})
    method_label = method_config.get("label", method_key)
    unit_price = method_config.get("unit_price_krw_per_m")
    length_m = float(length_mm) / 1000.0

    if not isinstance(unit_price, (int, float)) or unit_price <= 0:
        return {
            "method_key": method_key,
            "method_label": method_label,
            "length_m": length_m,
            "unit_price_krw_per_m": None,
            "estimated_cost_krw": None,
            "status": "공법별 단가 미설정",
        }

    return {
        "method_key": method_key,
        "method_label": method_label,
        "length_m": length_m,
        "unit_price_krw_per_m": float(unit_price),
        "estimated_cost_krw": length_m * float(unit_price),
        "status": "참고용 예상 금액",
    }


def save_capture(image, measurements):
    """현재 합성 화면과 같은 시점의 측정값을 PNG와 JSON으로 저장합니다."""
    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    captured_at = datetime.now()
    filename_stem = captured_at.strftime("crack_%Y%m%d_%H%M%S_%f")[:-3]
    image_path = CAPTURE_DIR / f"{filename_stem}.png"
    json_path = CAPTURE_DIR / f"{filename_stem}.json"

    encoded_ok, encoded_image = cv2.imencode(".png", image)
    if not encoded_ok:
        raise RuntimeError("PNG 인코딩에 실패했습니다.")

    # np.ndarray.tofile은 한글이 포함된 Windows 경로에서도 안전하게 저장됩니다.
    encoded_image.tofile(str(image_path))

    metadata = {
        "captured_at": captured_at.isoformat(timespec="milliseconds"),
        "image_file": image_path.name,
        "measurements": measurements,
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    return image_path, json_path


def thin_binary_mask(binary_mask):
    """Zhang-Suen 방식으로 마스크를 1픽셀 두께 중심선으로 만듭니다."""
    image = np.pad((binary_mask > 0).astype(np.uint8), 1)

    while True:
        changed = False

        for step in (0, 1):
            core = image[1:-1, 1:-1]
            p2 = image[:-2, 1:-1]
            p3 = image[:-2, 2:]
            p4 = image[1:-1, 2:]
            p5 = image[2:, 2:]
            p6 = image[2:, 1:-1]
            p7 = image[2:, :-2]
            p8 = image[1:-1, :-2]
            p9 = image[:-2, :-2]

            neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )

            if step == 0:
                directional_condition = (
                    (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
                )
            else:
                directional_condition = (
                    (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
                )

            remove = (
                (core == 1)
                & (neighbors >= 2)
                & (neighbors <= 6)
                & (transitions == 1)
                & directional_condition
            )

            if np.any(remove):
                core[remove] = 0
                changed = True

        if not changed:
            break

    return (image[1:-1, 1:-1] * 255).astype(np.uint8)


def calculate_skeleton_length_mm(
    skeleton, depth_mm, fx, fy, marker_mm_per_pixel=None
):
    """중심선 픽셀 연결을 실제 길이(mm)로 합산합니다."""
    pixels = skeleton > 0
    if marker_mm_per_pixel is not None:
        scale_x = float(marker_mm_per_pixel)
        scale_y = float(marker_mm_per_pixel)
    else:
        scale_x = float(depth_mm) / float(fx)
        scale_y = float(depth_mm) / float(fy)

    horizontal_edges = np.count_nonzero(pixels[:, :-1] & pixels[:, 1:])
    vertical_edges = np.count_nonzero(pixels[:-1, :] & pixels[1:, :])

    diagonal_down_right = (
        pixels[:-1, :-1]
        & pixels[1:, 1:]
        & ~pixels[:-1, 1:]
        & ~pixels[1:, :-1]
    )
    diagonal_down_left = (
        pixels[:-1, 1:]
        & pixels[1:, :-1]
        & ~pixels[:-1, :-1]
        & ~pixels[1:, 1:]
    )
    diagonal_edges = np.count_nonzero(diagonal_down_right) + np.count_nonzero(
        diagonal_down_left
    )

    return (
        horizontal_edges * scale_x
        + vertical_edges * scale_y
        + diagonal_edges * np.hypot(scale_x, scale_y)
    )

# ============================================================
# 1. 균열 종합 수치 측정 (길이, 최대 폭, 면적)
# ============================================================
def calculate_crack_metrics(
    binary_mask, depth_image, fx, fy, marker_mm_per_pixel=None
):
    """
    YOLO 마스크를 기반으로 균열의 2D 길이, 최대 폭, 면적을 구하고
    카메라 내부 파라미터(fx, fy)와 Depth 중앙값을 이용해 3D 실제 수치(mm)로 변환
    """
    # 1. 외곽선 추출
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, 0.0, None, np.zeros_like(binary_mask), "unavailable"
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 2. 2D 픽셀 수치 계산
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

    # 2-1. 균열 마스크의 모든 가지를 포함하는 중심선 생성
    skeleton = thin_binary_mask(binary_mask)
    
    # 2-2. 최대 폭 (Distance Transform을 이용해 마스크 내부에 그려지는 가장 큰 원의 지름 계산)
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    _, max_val, _, _ = cv2.minMaxLoc(dist_transform)
    pixel_max_width_2d = max_val * 2.0 
    
    # 2-3. 면적 (마스크 내부 픽셀 개수)
    pixel_area = cv2.contourArea(largest_contour)
    
    # 3. 균열 바닥이 아닌 주변 정상 벽면의 Depth로 mm/픽셀 비율 산출
    scale_kernel = np.ones((WALL_KERNEL_SIZE, WALL_KERNEL_SIZE), dtype=np.uint8)
    surrounding_mask = cv2.subtract(
        cv2.dilate(binary_mask, scale_kernel, iterations=1), binary_mask
    )
    surrounding_depths = depth_image[surrounding_mask > 0]
    valid_depths = surrounding_depths[surrounding_depths > 0]

    # 화면 가장자리 등에서 주변 벽면이 부족하면 마스크 내부값을 보조로 사용
    if valid_depths.size < 20:
        crack_depths = depth_image[binary_mask > 0]
        valid_depths = crack_depths[crack_depths > 0]

    if marker_mm_per_pixel is None and valid_depths.size == 0:
        return 0.0, 0.0, 0.0, approx_contour, skeleton, "unavailable"

    median_depth = float(np.median(valid_depths)) if valid_depths.size else 0.0
    
    # 4. 중심선과 카메라 내부 파라미터를 이용해 실제 길이 계산
    f_mean = (fx + fy) / 2.0
    real_length_mm = calculate_skeleton_length_mm(
        skeleton,
        median_depth,
        fx,
        fy,
        marker_mm_per_pixel=marker_mm_per_pixel,
    )

    if marker_mm_per_pixel is not None:
        scale_source = "aruco"
        real_width_mm = pixel_max_width_2d * marker_mm_per_pixel
        real_area_mm2 = pixel_area * marker_mm_per_pixel**2
    else:
        scale_source = "depth_approximation"
        real_width_mm = pixel_max_width_2d * (median_depth / f_mean)
        real_area_mm2 = (
            pixel_area * (median_depth / fx) * (median_depth / fy)
        )

    return (
        real_length_mm,
        real_width_mm,
        real_area_mm2,
        approx_contour,
        skeleton,
        scale_source,
    )

# ============================================================
# 2. 깊이(단차) 측정 (3D 평면 추정 방식)
# ============================================================
def calculate_crack_depth_plane_fitting(binary_mask, depth_image, fx, fy, cx, cy):
    mask_pixel_count = int(np.count_nonzero(binary_mask))
    if mask_pixel_count == 0:
        return {
            "depth_mm": None,
            "valid_ratio": 0.0,
            "recessed_ratio": 0.0,
            "noise_mm": None,
            "threshold_mm": None,
            "status": "빈 균열 마스크",
        }

    crack_ys, crack_xs = np.where(binary_mask > 0)
    crack_depths = depth_image[crack_ys, crack_xs]
    crack_valid = crack_depths > 0
    valid_ratio = float(np.count_nonzero(crack_valid) / mask_pixel_count)

    kernel = np.ones((WALL_KERNEL_SIZE, WALL_KERNEL_SIZE), dtype=np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    wall_mask = cv2.subtract(dilated_mask, binary_mask)

    wall_ys, wall_xs = np.where(wall_mask > 0)
    wall_depths = depth_image[wall_ys, wall_xs]
    wall_valid = wall_depths > 0
    wall_xs = wall_xs[wall_valid]
    wall_ys = wall_ys[wall_valid]
    wall_depths = wall_depths[wall_valid]

    if wall_depths.size < 20:
        return {
            "depth_mm": None,
            "valid_ratio": valid_ratio,
            "recessed_ratio": 0.0,
            "noise_mm": None,
            "threshold_mm": None,
            "status": "주변 벽면 Depth 부족",
        }

    wall_points_3d = deproject_pixels_to_points_mm(
        wall_xs, wall_ys, wall_depths, fx, fy, cx, cy
    )

    centroid = np.mean(wall_points_3d, axis=0)
    centered_points = wall_points_3d - centroid

    # Nx3 행렬 전체를 SVD하는 대신 동등한 3x3 공분산 행렬의
    # 최소 고유벡터를 사용해 벽면 법선을 빠르게 구합니다.
    covariance = centered_points.T @ centered_points
    _, eigenvectors = np.linalg.eigh(covariance)
    normal_vector = eigenvectors[:, 0]

    # 법선의 Z 방향을 카메라에서 멀어지는 방향으로 통일합니다.
    if normal_vector[2] < 0:
        normal_vector = -normal_vector

    wall_distances = centered_points @ normal_vector
    wall_distance_median = np.median(wall_distances)
    wall_noise_mm = 1.4826 * np.median(
        np.abs(wall_distances - wall_distance_median)
    )
    signal_threshold_mm = max(
        MIN_DEPTH_SIGNAL_MM,
        DEPTH_NOISE_MULTIPLIER * float(wall_noise_mm),
    )

    crack_xs = crack_xs[crack_valid]
    crack_ys = crack_ys[crack_valid]
    crack_depths = crack_depths[crack_valid]

    if crack_depths.size < 5:
        return {
            "depth_mm": None,
            "valid_ratio": valid_ratio,
            "recessed_ratio": 0.0,
            "noise_mm": float(wall_noise_mm),
            "threshold_mm": signal_threshold_mm,
            "status": "유효 Depth 픽셀 부족",
        }

    crack_points_3d = deproject_pixels_to_points_mm(
        crack_xs, crack_ys, crack_depths, fx, fy, cx, cy
    )

    vectors_to_crack = crack_points_3d - centroid
    signed_distances = vectors_to_crack @ normal_vector
    recessed_ratio = float(
        np.count_nonzero(signed_distances > signal_threshold_mm)
        / signed_distances.size
    )
    estimated_depth_mm = float(
        np.percentile(signed_distances, DEPTH_PERCENTILE)
    )

    visualization_data = {
        "pixel_xs": crack_xs,
        "pixel_ys": crack_ys,
        "signed_distances_mm": signed_distances,
    }

    if valid_ratio < MIN_DEPTH_VALID_RATIO:
        return {
            "depth_mm": None,
            "valid_ratio": valid_ratio,
            "recessed_ratio": recessed_ratio,
            "noise_mm": float(wall_noise_mm),
            "threshold_mm": signal_threshold_mm,
            "status": "균열 내부 Depth 부족",
            **visualization_data,
        }

    if (
        recessed_ratio < MIN_RECESSED_PIXEL_RATIO
        or estimated_depth_mm <= signal_threshold_mm
    ):
        return {
            "depth_mm": None,
            "valid_ratio": valid_ratio,
            "recessed_ratio": recessed_ratio,
            "noise_mm": float(wall_noise_mm),
            "threshold_mm": signal_threshold_mm,
            "status": "벽면 대비 깊이 신호 부족",
            **visualization_data,
        }

    return {
        "depth_mm": estimated_depth_mm,
        "valid_ratio": valid_ratio,
        "recessed_ratio": recessed_ratio,
        "noise_mm": float(wall_noise_mm),
        "threshold_mm": signal_threshold_mm,
        "status": "측정 가능",
        **visualization_data,
    }

# ============================================================
# YOLO & DepthAI 초기화
# ============================================================
print("YOLO 모델 로딩 중...")
torch.set_num_threads(TORCH_THREADS)
model = YOLO(MODEL_PATH)
print(f"YOLO 모델 로드 완료 (PyTorch CPU 스레드: {torch.get_num_threads()})")

repair_cost_config = load_repair_cost_config()
aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_detector = cv2.aruco.ArucoDetector(
    aruco_dictionary, cv2.aruco.DetectorParameters()
)

pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setPreviewSize(WIDTH, HEIGHT)
cam_rgb.setInterleaved(False)
cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
cam_rgb.setFps(FPS)

left = pipeline.create(dai.node.MonoCamera)
left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
left.setFps(FPS)

right = pipeline.create(dai.node.MonoCamera)
right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
right.setFps(FPS)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(True)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
stereo.setOutputSize(WIDTH, HEIGHT)

config = stereo.initialConfig.get()
# 좁은 균열의 미측정 영역을 주변 벽면 Depth로 채우지 않도록
# 진단 중에는 보정 필터를 끄고 원본에 가까운 Depth를 사용합니다.
config.postProcessing.speckleFilter.enable = False
config.postProcessing.spatialFilter.enable = False
config.postProcessing.temporalFilter.enable = False
stereo.initialConfig.set(config)

left.out.link(stereo.left)
right.out.link(stereo.right)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

# ============================================================
# 메인 루프 실행
# ============================================================
print("\nOAK-D Lite 시작 (종료: q)\n")
print("Depth 진단 모드: 보정 및 hole-filling 필터 OFF")

frame_count = 0
smoothed_fps = 0.0
marker_mm_per_pixel = None
marker_corners = None
marker_id = None

with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER) as device:
    print(f"USB 연결 속도: {device.getUsbSpeed()}")
    q_rgb = device.getOutputQueue(name="rgb", maxSize=2, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=2, blocking=False)

    calib = device.readCalibration()
    intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, WIDTH, HEIGHT)
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]

    while True:
        frame_started_at = time.perf_counter()
        rgb_packet = q_rgb.get()
        depth_packet = q_depth.get()
        frame_count += 1

        color_image = rgb_packet.getCvFrame()
        depth_image = depth_packet.getFrame()

        if color_image.shape[:2] != (HEIGHT, WIDTH):
            color_image = cv2.resize(color_image, (WIDTH, HEIGHT))
        if depth_image.shape[:2] != (HEIGHT, WIDTH):
            depth_image = cv2.resize(depth_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

        if frame_count == 1 or frame_count % ARUCO_DETECT_INTERVAL == 0:
            (
                detected_scale,
                detected_corners,
                detected_marker_id,
            ) = detect_aruco_scale(color_image, aruco_detector)
            marker_mm_per_pixel = detected_scale
            marker_corners = detected_corners
            marker_id = detected_marker_id

        inference_started_at = time.perf_counter()
        results = model(
            color_image,
            imgsz=YOLO_IMGSZ,
            conf=CONF_THRESHOLD,
            verbose=False,
        )
        inference_ms = (time.perf_counter() - inference_started_at) * 1000.0
        result = results[0]
        annotated_frame = result.plot()

        if marker_corners is not None:
            marker_points = marker_corners.astype(np.int32)
            cv2.polylines(
                annotated_frame, [marker_points], True, (0, 255, 0), 2
            )
            cv2.putText(
                annotated_frame,
                f"ArUco {marker_id}: {marker_mm_per_pixel:.3f} mm/px",
                tuple(marker_points[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                annotated_frame,
                "Scale: Depth approximation (place 50mm ArUco #0)",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                2,
            )
        crack_roi_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        local_depth_valid_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
        local_depth_values = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        crack_contours = []

        crack_count = 0
        if result.masks is not None and result.boxes is not None:
            crack_count = sum(
                result.names[int(class_id)].lower() == "crack"
                for class_id in result.boxes.cls.tolist()
            )
        cv2.putText(annotated_frame, f"Cracks: {crack_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if result.masks is not None:
            for idx, polygon in enumerate(result.masks.xy):
                class_id = int(result.boxes.cls[idx].item())
                if result.names[class_id].lower() != "crack":
                    continue

                pts = np.array(polygon, dtype=np.int32)
                if len(pts) < 3: continue

                pts[:, 0] = np.clip(pts[:, 0], 0, WIDTH - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, HEIGHT - 1)

                binary_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                cv2.fillPoly(binary_mask, [pts], 255)
                crack_roi_mask |= binary_mask > 0
                crack_contours.append(pts)

                # 새로운 종합 수치 계산 (길이, 최대폭, 면적)
                (
                    crack_length,
                    crack_width,
                    crack_area,
                    approx_contour,
                    crack_skeleton,
                    scale_source,
                ) = calculate_crack_metrics(
                    binary_mask,
                    depth_image,
                    fx,
                    fy,
                    marker_mm_per_pixel=marker_mm_per_pixel,
                )

                repair_result = estimate_repair_cost(
                    crack_length, crack_width, repair_cost_config
                )
                
                # 단차(깊이) 계산
                depth_result = calculate_crack_depth_plane_fitting(
                    binary_mask, depth_image, fx, fy, cx, cy
                )
                crack_depth = depth_result["depth_mm"]

                depth_xs = depth_result.get("pixel_xs")
                depth_ys = depth_result.get("pixel_ys")
                signed_depths = depth_result.get("signed_distances_mm")
                if (
                    depth_xs is not None
                    and depth_ys is not None
                    and signed_depths is not None
                ):
                    clipped_depths = np.clip(
                        signed_depths, 0.0, LOCAL_DEPTH_VIS_MAX_MM
                    )
                    previous_depths = local_depth_values[depth_ys, depth_xs]
                    local_depth_values[depth_ys, depth_xs] = np.maximum(
                        previous_depths, clipped_depths
                    )
                    local_depth_valid_mask[depth_ys, depth_xs] = True

                # ----------------------------------------------------
                # DB 저장용 터미널 출력 (JSON 포맷) - 30프레임에 한 번씩만 출력
                # ----------------------------------------------------
                if frame_count % 30 == 0 and crack_length > 0:
                    report_data = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "defect": "Crack",
                        "confidence": round(float(result.boxes.conf[idx].item()), 2),
                        "length_mm": round(float(crack_length), 1),
                        "length_method": "skeleton_total_length",
                        "dimension_scale_source": scale_source,
                        "aruco_marker_size_mm": ARUCO_MARKER_SIZE_MM if scale_source == "aruco" else None,
                        "max_width_mm": round(float(crack_width), 2),
                        "area_mm2": round(float(crack_area), 1),
                        "recommended_repair_method": repair_result["method_label"],
                        "repair_quantity_m": round(repair_result["length_m"], 3),
                        "unit_price_krw_per_m": repair_result["unit_price_krw_per_m"],
                        "estimated_repair_cost_krw": round(repair_result["estimated_cost_krw"]) if repair_result["estimated_cost_krw"] is not None else None,
                        "cost_status": repair_result["status"],
                        "depth_diff_mm": round(float(crack_depth), 1) if crack_depth is not None else None,
                        "depth_reliable": crack_depth is not None,
                        "depth_valid_ratio": round(depth_result["valid_ratio"], 3),
                        "depth_recessed_ratio": round(depth_result["recessed_ratio"], 3),
                        "depth_noise_mm": round(depth_result["noise_mm"], 2) if depth_result["noise_mm"] is not None else None,
                        "depth_threshold_mm": round(depth_result["threshold_mm"], 2) if depth_result["threshold_mm"] is not None else None,
                        "depth_status": depth_result["status"],
                        "risk_level": (
                            "판정 보류 (Depth 신뢰도 부족)"
                            if crack_depth is None
                            else "WARNING (c등급 후보)"
                            if crack_width >= CRITICAL_WIDTH_MM
                            else "NORMAL (a등급 후보)"
                        )
                    }
                    print(json.dumps(report_data, ensure_ascii=False, indent=2))

                # ----------------------------------------------------
                # 화면 표시 (OpenCV)
                # ----------------------------------------------------
                if approx_contour is not None:
                    # Depth 신뢰도가 없으면 노란색으로 판정 보류를 표시합니다.
                    color = (
                        (0, 255, 255)
                        if crack_depth is None
                        else (0, 0, 255)
                        if crack_width >= CRITICAL_WIDTH_MM
                        else (255, 0, 0)
                    )
                    cv2.drawContours(annotated_frame, [approx_contour], -1, color, 2)

                # 실제 길이 합산에 사용된 1픽셀 중심선을 초록색으로 표시
                annotated_frame[crack_skeleton > 0] = (0, 255, 0)

                x_min, y_max = int(np.min(pts[:, 0])), int(np.max(pts[:, 1]))
                x_text, y_text = max(5, x_min), min(HEIGHT - 80, y_max + 20)

                # 텍스트 포맷팅
                t_len = f"L(total): {crack_length:.1f}mm" if crack_length > 0 else "L: N/A"
                t_wid = f"W: {crack_width:.2f}mm" if crack_width > 0 else "W: N/A"
                t_area = f"A: {crack_area:.0f}mm2" if crack_area > 0 else "A: N/A"
                t_cost = (
                    f"COST: {repair_result['estimated_cost_krw']:,.0f} KRW"
                    if repair_result["estimated_cost_krw"] is not None
                    else "COST: set unit price in repair_cost_config.json"
                )
                t_dep = (
                    f"D: {crack_depth:.1f}mm V:{depth_result['valid_ratio']:.0%} S:{depth_result['recessed_ratio']:.0%}"
                    if crack_depth is not None
                    else f"D: N/A V:{depth_result['valid_ratio']:.0%} S:{depth_result['recessed_ratio']:.0%}"
                )

                # 화면에 정보 출력 (폭이 0.3 초과면 글씨 색을 빨갛게 강조)
                text_color = (0, 0, 255) if crack_width >= CRITICAL_WIDTH_MM else (0, 255, 255)
                
                cv2.putText(annotated_frame, t_len, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(annotated_frame, t_wid, (x_text, y_text + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                cv2.putText(annotated_frame, t_area, (x_text, y_text + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                depth_text_color = (0, 165, 255) if crack_depth is not None else (160, 160, 160)
                cv2.putText(annotated_frame, t_dep, (x_text, y_text + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, depth_text_color, 2)
                cv2.putText(annotated_frame, t_cost, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 주변 벽면을 0mm로 둔 균열 전용 국부 깊이 맵입니다.
        local_depth_normalized = np.clip(
            local_depth_values / LOCAL_DEPTH_VIS_MAX_MM * 255.0,
            0,
            255,
        ).astype(np.uint8)
        local_depth_colormap = cv2.applyColorMap(
            local_depth_normalized, cv2.COLORMAP_JET
        )
        local_depth_colormap[~crack_roi_mask] = 0
        local_depth_colormap[
            crack_roi_mask & ~local_depth_valid_mask
        ] = (80, 80, 80)

        for contour in crack_contours:
            cv2.polylines(
                local_depth_colormap, [contour], True, (255, 255, 255), 1
            )

        cv2.putText(
            local_depth_colormap,
            f"LOCAL CRACK DEPTH 0-{LOCAL_DEPTH_VIS_MAX_MM:.0f} mm",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            local_depth_colormap,
            "BLUE=surface  RED=deep  GRAY=invalid",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        combined = np.hstack((annotated_frame, local_depth_colormap))

        frame_seconds = time.perf_counter() - frame_started_at
        current_fps = 1.0 / frame_seconds if frame_seconds > 0 else 0.0
        smoothed_fps = (
            current_fps
            if smoothed_fps == 0.0
            else 0.9 * smoothed_fps + 0.1 * current_fps
        )
        cv2.putText(
            combined,
            f"FPS: {smoothed_fps:.1f} | Inference: {inference_ms:.0f} ms",
            (10, HEIGHT - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imshow("OAK-D Lite Crack Detection", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
