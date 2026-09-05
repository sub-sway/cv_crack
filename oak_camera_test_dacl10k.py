import depthai as dai
import numpy as np
import cv2
import json
from datetime import datetime
from ultralytics import YOLO

# ============================================================
# 설정
# ============================================================
MODEL_PATH = "runs/segment/runs/dacl10k/yolo11n_seg_1024-2/weights/best.pt"

WIDTH = 640
HEIGHT = 480
FPS = 30

CONF_THRESHOLD = 0.15
YOLO_IMGSZ = 1024
WALL_KERNEL_SIZE = 15

# 위험 기준 (교량 점검 기준서 기반)
CRITICAL_WIDTH_MM = 0.3 

# ============================================================
# 헬퍼 함수
# ============================================================
def deproject_pixel_to_point_mm(x, y, depth_mm, fx, fy, cx, cy):
    Z = float(depth_mm)
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    return np.array([X, Y, Z], dtype=np.float32)

# ============================================================
# 1. 균열 종합 수치 측정 (길이, 최대 폭, 면적)
# ============================================================
def calculate_crack_metrics(binary_mask, depth_image, fx, fy):
    """
    YOLO 마스크를 기반으로 균열의 2D 길이, 최대 폭, 면적을 구하고
    카메라 내부 파라미터(fx, fy)와 Depth 중앙값을 이용해 3D 실제 수치(mm)로 변환
    """
    # 1. 외곽선 추출
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0, 0.0, None
        
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 2. 2D 픽셀 수치 계산
    # 2-1. 길이 (외곽선을 단순화하여 둘레의 절반을 길이로 산정)
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    pixel_length_2d = cv2.arcLength(approx_contour, True) / 2.0
    
    # 2-2. 최대 폭 (Distance Transform을 이용해 마스크 내부에 그려지는 가장 큰 원의 지름 계산)
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    _, max_val, _, _ = cv2.minMaxLoc(dist_transform)
    pixel_max_width_2d = max_val * 2.0 
    
    # 2-3. 면적 (마스크 내부 픽셀 개수)
    pixel_area = cv2.contourArea(largest_contour)
    
    # 3. Depth 중앙값 추출 (노이즈 방지)
    crack_depths = depth_image[binary_mask > 0]
    valid_depths = crack_depths[crack_depths > 0]
    if len(valid_depths) == 0:
        return 0.0, 0.0, 0.0, approx_contour
        
    median_depth = np.median(valid_depths)
    
    # 4. 2D -> 3D 변환 (카메라 비례식)
    f_mean = (fx + fy) / 2.0
    
    real_length_mm = pixel_length_2d * (median_depth / f_mean)
    real_width_mm = pixel_max_width_2d * (median_depth / f_mean)
    real_area_mm2 = pixel_area * (median_depth / fx) * (median_depth / fy) # 가로세로 비율 각각 적용
    
    return real_length_mm, real_width_mm, real_area_mm2, approx_contour

# ============================================================
# 2. 깊이(단차) 측정 (3D 평면 추정 방식)
# ============================================================
def calculate_crack_depth_plane_fitting(binary_mask, depth_image, fx, fy, cx, cy):
    kernel = np.ones((WALL_KERNEL_SIZE, WALL_KERNEL_SIZE), dtype=np.uint8)
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
    wall_mask = cv2.subtract(dilated_mask, binary_mask)

    wall_ys, wall_xs = np.where(wall_mask > 0)
    wall_points_3d = [deproject_pixel_to_point_mm(x, y, depth_image[y, x], fx, fy, cx, cy) 
                      for x, y in zip(wall_xs, wall_ys) if depth_image[y, x] > 0]
    
    if len(wall_points_3d) < 20: return None
    wall_points_3d = np.array(wall_points_3d)

    centroid = np.mean(wall_points_3d, axis=0)
    centered_points = wall_points_3d - centroid
    _, _, vh = np.linalg.svd(centered_points)
    normal_vector = vh[2, :] 

    crack_ys, crack_xs = np.where(binary_mask > 0)
    crack_points_3d = [deproject_pixel_to_point_mm(x, y, depth_image[y, x], fx, fy, cx, cy) 
                       for x, y in zip(crack_xs, crack_ys) if depth_image[y, x] > 0]
            
    if len(crack_points_3d) < 5: return None
    crack_points_3d = np.array(crack_points_3d)

    vectors_to_crack = crack_points_3d - centroid
    distances = np.abs(np.dot(vectors_to_crack, normal_vector))
    
    return np.median(distances)

# ============================================================
# YOLO & DepthAI 초기화
# ============================================================
print("YOLO 모델 로딩 중...")
model = YOLO(MODEL_PATH)
print("YOLO 모델 로드 완료")

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
config.postProcessing.speckleFilter.enable = True
config.postProcessing.speckleFilter.speckleRange = 50
config.postProcessing.spatialFilter.enable = True
config.postProcessing.spatialFilter.holeFillingRadius = 2
config.postProcessing.spatialFilter.numIterations = 1
config.postProcessing.spatialFilter.alpha = 0.5
config.postProcessing.spatialFilter.delta = 20
config.postProcessing.temporalFilter.enable = True
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

frame_count = 0

with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.HIGH) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=2, blocking=False)
    q_depth = device.getOutputQueue(name="depth", maxSize=2, blocking=False)

    calib = device.readCalibration()
    intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, WIDTH, HEIGHT)
    fx, fy = intrinsics[0][0], intrinsics[1][1]
    cx, cy = intrinsics[0][2], intrinsics[1][2]

    while True:
        rgb_packet = q_rgb.get()
        depth_packet = q_depth.get()
        frame_count += 1

        color_image = rgb_packet.getCvFrame()
        depth_image = depth_packet.getFrame()

        if color_image.shape[:2] != (HEIGHT, WIDTH):
            color_image = cv2.resize(color_image, (WIDTH, HEIGHT))
        if depth_image.shape[:2] != (HEIGHT, WIDTH):
            depth_image = cv2.resize(depth_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)

        results = model(color_image, imgsz=YOLO_IMGSZ, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]
        annotated_frame = result.plot()

        crack_count = len(result.masks.xy) if result.masks is not None else 0
        cv2.putText(annotated_frame, f"Cracks: {crack_count}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if result.masks is not None:
            for idx, polygon in enumerate(result.masks.xy):
                pts = np.array(polygon, dtype=np.int32)
                if len(pts) < 3: continue

                pts[:, 0] = np.clip(pts[:, 0], 0, WIDTH - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, HEIGHT - 1)

                binary_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                cv2.fillPoly(binary_mask, [pts], 255)

                # 새로운 종합 수치 계산 (길이, 최대폭, 면적)
                crack_length, crack_width, crack_area, approx_contour = calculate_crack_metrics(binary_mask, depth_image, fx, fy)
                
                # 단차(깊이) 계산
                crack_depth = calculate_crack_depth_plane_fitting(binary_mask, depth_image, fx, fy, cx, cy)

                # ----------------------------------------------------
                # DB 저장용 터미널 출력 (JSON 포맷) - 30프레임에 한 번씩만 출력
                # ----------------------------------------------------
                if frame_count % 30 == 0 and crack_length > 0:
                    report_data = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "defect": "Crack",
                        "confidence": round(float(result.boxes.conf[idx].item()), 2),
                        "length_mm": round(crack_length, 1),
                        "max_width_mm": round(crack_width, 2),
                        "area_mm2": round(crack_area, 1),
                        "depth_diff_mm": round(crack_depth, 1) if crack_depth else None,
                        "risk_level": "WARNING (c등급 후보)" if crack_width >= CRITICAL_WIDTH_MM else "NORMAL (a등급 후보)"
                    }
                    print(json.dumps(report_data, ensure_ascii=False, indent=2))

                # ----------------------------------------------------
                # 화면 표시 (OpenCV)
                # ----------------------------------------------------
                if approx_contour is not None:
                    # 폭이 0.3mm 이상이면 빨간색(위험), 아니면 파란색 테두리
                    color = (0, 0, 255) if crack_width >= CRITICAL_WIDTH_MM else (255, 0, 0)
                    cv2.drawContours(annotated_frame, [approx_contour], -1, color, 2)

                x_min, y_max = int(np.min(pts[:, 0])), int(np.max(pts[:, 1]))
                x_text, y_text = max(5, x_min), min(HEIGHT - 80, y_max + 20)

                # 텍스트 포맷팅
                t_len = f"L: {crack_length:.1f}mm" if crack_length > 0 else "L: N/A"
                t_wid = f"W: {crack_width:.2f}mm" if crack_width > 0 else "W: N/A"
                t_area = f"A: {crack_area:.0f}mm2" if crack_area > 0 else "A: N/A"
                t_dep = f"D: {crack_depth:.1f}mm" if crack_depth is not None else "D: N/A"

                # 화면에 정보 출력 (폭이 0.3 초과면 글씨 색을 빨갛게 강조)
                text_color = (0, 0, 255) if crack_width >= CRITICAL_WIDTH_MM else (0, 255, 255)
                
                cv2.putText(annotated_frame, t_len, (x_text, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(annotated_frame, t_wid, (x_text, y_text + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                cv2.putText(annotated_frame, t_area, (x_text, y_text + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(annotated_frame, t_dep, (x_text, y_text + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        # Depth 맵 시각화
        valid_mask = depth_image > 0
        depth_normalized = np.zeros(depth_image.shape, dtype=np.uint8)

        if np.any(valid_mask):
            valid_depths = depth_image[valid_mask]
            min_depth, max_depth = np.percentile(valid_depths, 5), np.percentile(valid_depths, 95)
            if max_depth > min_depth:
                normalized = (depth_image.astype(np.float32) - min_depth) / (max_depth - min_depth) * 255
                depth_normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        depth_normalized[~valid_mask] = 0
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        depth_colormap[~valid_mask] = 0

        combined = np.hstack((annotated_frame, depth_colormap))
        cv2.imshow("OAK-D Lite Crack Detection", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()