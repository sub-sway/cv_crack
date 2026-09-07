"""OAK-D Lite + YOLO11-seg 기반 터널 균열 실시간 모니터링.

실행:
    python crack_monitor.py                  # 기본 (로컬 창 표시)
    python crack_monitor.py --no-window      # 헤드리스 (웹 대시보드만)
    python crack_monitor.py --port 8080 --conf 0.4

웹 대시보드: http://<PC IP>:5000
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, request, send_from_directory
from ultralytics import YOLO

import depthai as dai
from observation_store import ObservationStore, ObservationTracker
from repair_costs import METHODS, estimate_repair_cost, present_record, summarize_costs
from estimate_setup import EstimateSetupStore, DEFAULT_INPUTS, DEPTH_SCENARIOS_CM, estimate_depth_scenarios
from width_observations import WIDTH_VERSION
from imu_gateway import read_imu_state


BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 설정
# ============================================================

@dataclass
class Settings:
    # 모델 / 카메라
    model_path: str = "runs/segment/runs/dacl10k/yolo11n_seg_1024-2/weights/best.pt"
    # OAK-D-Lite의 IMX214는 1080p에서 16:9입니다. 여기에 4:3(640x480)을 요구하면
    # preview는 센서 가로의 75%만 중앙 크롭하는 반면 CAM_A 정렬 depth는 전체
    # FOV를 그대로 담기 때문에, 두 영상이 가로로 최대 80px 어긋납니다.
    # 16:9를 유지하면 크롭이 사라져 마스크 좌표와 depth 화소가 1:1로 맞습니다.
    width: int = 640
    height: int = 360
    fps: int = 30
    conf_threshold: float = 0.35
    yolo_imgsz: int = 640
    torch_threads: int = 6

    # 고해상도 폭 측정
    # preview 640px는 1m에서 2.15mm/px라 폭 측정한계가 4.3mm입니다. RC 등급
    # 경계(0.1~1.0mm)보다 한 자릿수 크므로 폭 판정에 쓸 수 없습니다. 탐지는
    # preview로 하고, 폭·길이는 같은 ISP의 1080p 프레임에서 다시 잽니다.
    use_highres_width: bool = True
    highres_width: int = 1920
    highres_height: int = 1080
    highres_margin_px: int = 24     # 고해상도 ROI 여유
    highres_block_px: int = 51      # 적응형 임계 블록(홀수)
    highres_bias: int = 30          # 적응형 임계 offset (클수록 보수적)
    highres_seed_dilate_px: int = 9 # YOLO 마스크를 넓힌 이 범위 안에서만 재분할
    highres_area_guard: float = 3.0 # 재분할 면적이 씨앗의 이 배를 넘으면 폐기
    highres_max_seq_lag: int = 2    # preview와 이만큼 넘게 어긋나면 사용 안 함
    # 임계 기반 재분할은 렌즈 블러 경계를 함께 먹어 약 2px 과대측정합니다.
    # 합성 검증에서 4px 이상이면 오차 10% 이내였고 3px는 +100%였으므로,
    # 고해상도 측정의 판정 한계는 이론값(2px)이 아니라 4px로 둡니다.
    highres_limit_px: float = 4.0

    # 마스크 / 벽면
    wall_kernel_size: int = 15      # 균열 주변 벽면 링 두께(px)
    roi_margin_px: int = 24         # ROI 여유 (벽면 링 + a)
    min_mask_pixels: int = 30       # 이보다 작은 마스크는 무시

    # Depth 신뢰도 판정
    min_depth_valid_ratio: float = 0.30
    min_recessed_pixel_ratio: float = 0.10
    min_depth_signal_mm: float = 5.0
    depth_noise_multiplier: float = 3.0
    depth_percentile: int = 90
    local_depth_vis_max_mm: float = 30.0

    # 치수 보정
    aruco_marker_size_mm: float = 50.0
    aruco_detect_interval: int = 5      # N프레임마다 재검출
    aruco_hold_seconds: float = 3.0     # 검출 실패 시 마지막 값 유지 시간

    # 이력 / 저장
    history_db_path: Path = BASE_DIR / "observations.sqlite3"
    history_min_interval_s: float = 3.0
    auto_capture_min_grade: str = "c"       # 이 등급 이상이면 자동으로 캡처
    auto_capture_cooldown_s: float = 10.0
    capture_dir: Path = BASE_DIR / "captures"
    repair_cost_config_path: Path = BASE_DIR / "repair_cost_config.json"
    inspection_config_path: Path = BASE_DIR / "inspection_config.json"

    # 서버 / 표시
    host: str = "0.0.0.0"
    port: int = 5000
    jpeg_quality: int = 85
    show_local_window: bool = True


S = Settings()
estimate_setup_store = EstimateSetupStore(BASE_DIR / "estimate_setup.json")


# ============================================================
# 점검 기준 (안전점검 상태평가 기준표)
# ============================================================

# 부재 대분류 → 포함 부재
MEMBER_GROUPS: dict[str, list[str]] = {
    "상부구조": ["바닥판", "거더", "가로보"],
    "하부구조": ["교각", "교대", "기초"],
    "연결·지지부": ["교량받침", "신축이음"],
    "부속·배수부": ["배수시설", "난간·연석", "교면포장"],
}

# 재질별 균열폭 등급 기준 (mm).
# (상한, 등급) — 폭이 상한 미만이면 그 등급. 어느 구간에도 안 들면 fallback 등급.
CRACK_GRADE_TABLES: dict[str, tuple[tuple[float, str], ...]] = {
    # 철근콘크리트 바닥판·거더·가로보·교각·교대 공통
    "rc": ((0.1, "a"), (0.3, "b"), (0.5, "c"), (1.0, "d")),
    # 프리스트레스 콘크리트 거더 (균열이 있으면 a는 불가)
    "psc": ((0.0, "a"), (0.2, "b"), (0.3, "c"), (0.5, "d")),
    # 콘크리트 가로보 (e등급 없음)
    "rc_crossbeam": ((0.1, "a"), (0.3, "b"), (0.5, "c")),
}

GRADE_FALLBACK: dict[str, str] = {"rc": "e", "psc": "e", "rc_crossbeam": "d"}

MATERIAL_LABELS: dict[str, str] = {
    "rc": "철근콘크리트",
    "psc": "프리스트레스 콘크리트",
    "rc_crossbeam": "콘크리트 가로보",
    "steel": "강재 (균열폭 기준 미적용)",
}

# 등급 → (예비등급, 권고 조치)
GRADE_ACTIONS: dict[str, tuple[str, str]] = {
    "a": ("양호", "조치 불필요"),
    "b": ("관찰", "주기적 관찰"),
    "c": ("보수 검토", "근접 확인 후 보수 계획 수립"),
    "d": ("정밀점검 우선", "정밀점검 및 보수·보강 검토"),
    "e": ("긴급 조치", "정밀안전진단 시행"),
}

GRADE_RISK_CODES: dict[str, str] = {
    "a": "normal",
    "b": "caution",
    "c": "danger",
    "d": "danger",
    "e": "danger",
}

GRADE_ORDER = "abcde"

DEFAULT_INSPECTION: dict[str, Any] = {
    "bridge_name": "",
    "bridge_id": "",
    "site": "",
    "inspector": "",
    "weather": "맑음",
    "capture_distance_mm": None,
    "inspection_scope": "",
    "member_group": "하부구조",
    "member": "교각",
    "member_label": "P1 교각",
    "member_material": "rc",
    "member_position": "전면 하단",
    "repair_method": "",
    "repair_review_note": "",
}


# ============================================================
# 프레임 브로커 (MJPEG 스트림용)
# ============================================================

class FrameBroker:
    """최신 프레임을 보관하고, 새 프레임이 생겼을 때만 소비자를 깨웁니다.

    같은 프레임을 여러 번 JPEG 인코딩하지 않도록 seq 단위로 캐시합니다.
    """

    def __init__(self, jpeg_quality: int = 85):
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._jpeg: bytes | None = None
        self._jpeg_seq = -1
        self._quality = jpeg_quality

    def publish(self, frame: np.ndarray) -> None:
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._cond.notify_all()

    def snapshot(self) -> np.ndarray | None:
        with self._cond:
            return None if self._frame is None else self._frame.copy()

    def jpeg(self, last_seq: int, timeout: float = 1.0) -> tuple[bytes | None, int]:
        """새 프레임을 기다렸다가 JPEG 바이트와 seq를 반환합니다."""
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)

            if self._frame is None or self._seq == last_seq:
                return None, last_seq

            if self._jpeg_seq == self._seq:
                return self._jpeg, self._seq

            frame = self._frame
            seq = self._seq

        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._quality]
        )
        if not ok:
            return None, seq

        data = buffer.tobytes()
        with self._cond:
            if seq >= self._jpeg_seq:
                self._jpeg, self._jpeg_seq = data, seq
        return data, seq


# ============================================================
# 공유 상태
# ============================================================

app = Flask(__name__, static_folder=str(BASE_DIR / "static"), static_url_path="/static")

broker = FrameBroker(S.jpeg_quality)
stop_event = threading.Event()

data_lock = threading.Lock()
history_lock = threading.Lock()

latest_data: dict[str, Any] = {
    "timestamp": None,
    "crack_count": 0,
    "detections": [],
    "fps": 0.0,
    "inference_ms": 0.0,
    "usb_speed": None,
    "aruco_detected": False,
    "aruco_marker_id": None,
    "aruco_scale_mm_per_pixel": None,
    "measurement_limit_mm": None,
    "camera_status": "starting",
    "error": None,
}

observation_store: ObservationStore | None = None


def get_observation_store() -> ObservationStore:
    global observation_store
    if observation_store is None:
        observation_store = ObservationStore(S.history_db_path)
    return observation_store

inspection_lock = threading.Lock()
inspection: dict[str, Any] = dict(DEFAULT_INSPECTION)

model: YOLO | None = None
repair_cost_config: dict[str, Any] = {"currency": "KRW", "methods": {}}
_repair_config_mtime: float | None = None


def set_status(**fields: Any) -> None:
    with data_lock:
        latest_data.update(fields)


# ============================================================
# 보수비 설정
# ============================================================

def load_repair_cost_config(force: bool = False) -> dict[str, Any]:
    """설정 파일을 읽고, 파일이 바뀌었으면 자동으로 다시 읽습니다."""
    global repair_cost_config, _repair_config_mtime

    path = S.repair_cost_config_path
    if not path.exists():
        repair_cost_config = {"currency": "KRW", "methods": {}}
        _repair_config_mtime = None
        return repair_cost_config

    mtime = path.stat().st_mtime
    if not force and mtime == _repair_config_mtime:
        return repair_cost_config

    try:
        with path.open("r", encoding="utf-8") as file:
            repair_cost_config = json.load(file)
        _repair_config_mtime = mtime
    except (json.JSONDecodeError, OSError) as error:
        print(f"[설정] repair_cost_config.json 읽기 실패: {error}")
        # 수정 중인 잘못된 설정으로 이전 단가를 계속 적용하지 않습니다.
        repair_cost_config = {"currency": "KRW", "methods": {}}
        _repair_config_mtime = None

    return repair_cost_config


# ============================================================
# ArUco 스케일 추적
# ============================================================

class ArucoScaleTracker:
    """ArUco 마커로 mm/픽셀 비율을 구하고, 짧은 미검출은 이전 값으로 버팁니다."""

    def __init__(self) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        # mm/px가 곧 균열 폭이므로 코너 위치 오차가 그대로 폭 오차가 됩니다.
        # 기본 검출은 코너를 정수 픽셀로만 잡아서, 50mm 마커가 40px로 보일 때
        # ±0.5px = ±1.3% 폭 오차가 납니다. 서브픽셀 보정으로 이를 줄입니다.
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        self.mm_per_pixel: float | None = None
        self.corners: np.ndarray | None = None
        self.marker_id: int | None = None
        self.is_stale = False
        self._updated_at = 0.0

    def update(self, color_image: np.ndarray, pixel_scale: float = 1.0) -> None:
        """pixel_scale은 (preview 픽셀 / 입력 영상 픽셀) 비율입니다.

        1080p 프레임에서 검출하면 마커가 3배 커져 코너 오차가 1/3로 줄고,
        결과 mm/px는 preview 좌표계로 환산해 보관합니다.
        """
        detected = self._detect(color_image, pixel_scale)

        if detected is not None:
            self.mm_per_pixel, self.corners, self.marker_id = detected
            self.is_stale = False
            self._updated_at = time.monotonic()
            return

        if self.mm_per_pixel is None:
            return

        elapsed = time.monotonic() - self._updated_at
        if elapsed > S.aruco_hold_seconds:
            self.mm_per_pixel = None
            self.corners = None
            self.marker_id = None
            self.is_stale = False
        else:
            self.is_stale = True
            self.corners = None  # 위치는 더 이상 유효하지 않음

    def _detect(
        self, color_image: np.ndarray, pixel_scale: float = 1.0
    ) -> tuple[float, np.ndarray, int] | None:
        corners, ids, _ = self._detector.detectMarkers(color_image)
        if ids is None or len(corners) == 0:
            return None

        areas = [abs(cv2.contourArea(corner.reshape(4, 2))) for corner in corners]
        index = int(np.argmax(areas))
        marker_corners = corners[index].reshape(4, 2)

        sides = [
            float(np.linalg.norm(marker_corners[(i + 1) % 4] - marker_corners[i]))
            for i in range(4)
        ]
        pixel_size = float(np.mean(sides))
        if pixel_size <= 1e-6:
            return None

        # 입력이 1080p이면 preview에서의 마커 크기는 pixel_size * pixel_scale 입니다.
        mm_per_pixel = S.aruco_marker_size_mm / (pixel_size * pixel_scale)
        marker_id = int(np.asarray(ids).reshape(-1)[index])
        return mm_per_pixel, marker_corners * pixel_scale, marker_id


# ============================================================
# 기하 계산 헬퍼
# ============================================================

def deproject_pixels_to_points_mm(
    xs: np.ndarray,
    ys: np.ndarray,
    depths_mm: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Depth 픽셀 묶음을 한 번의 NumPy 연산으로 3D 좌표(mm)로 변환합니다."""
    z = depths_mm.astype(np.float32, copy=False)
    x = (xs.astype(np.float32, copy=False) - cx) * z / fx
    y = (ys.astype(np.float32, copy=False) - cy) * z / fy
    return np.column_stack((x, y, z))


_HAS_XIMGPROC = hasattr(cv2, "ximgproc") and hasattr(
    getattr(cv2, "ximgproc", None), "thinning"
)


def thin_binary_mask(binary_mask: np.ndarray) -> np.ndarray:
    """마스크를 1픽셀 두께 중심선으로 만듭니다 (Zhang-Suen).

    opencv-contrib가 있으면 C++ 구현을 쓰고, 없으면 NumPy 폴백을 씁니다.
    폴백도 ROI 크기에서만 돌기 때문에 전체 프레임을 쓰던 이전 방식보다 훨씬 빠릅니다.
    """
    if _HAS_XIMGPROC:
        return cv2.ximgproc.thinning(
            (binary_mask > 0).astype(np.uint8) * 255,
            thinningType=cv2.ximgproc.THINNING_ZHANGSUEN,
        )

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

            ordered = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
            transitions = sum(
                ((ordered[i] == 0) & (ordered[i + 1] == 1)).astype(np.uint8)
                for i in range(8)
            )

            if step == 0:
                directional = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                directional = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)

            remove = (
                (core == 1)
                & (neighbors >= 2)
                & (neighbors <= 6)
                & (transitions == 1)
                & directional
            )

            if np.any(remove):
                core[remove] = 0
                changed = True

        if not changed:
            break

    return (image[1:-1, 1:-1] * 255).astype(np.uint8)


def skeleton_length_mm(
    skeleton: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> float:
    """중심선 픽셀 연결을 실제 길이(mm)로 합산합니다."""
    pixels = skeleton > 0
    if not pixels.any():
        return 0.0

    horizontal = np.count_nonzero(pixels[:, :-1] & pixels[:, 1:])
    vertical = np.count_nonzero(pixels[:-1, :] & pixels[1:, :])

    down_right = (
        pixels[:-1, :-1] & pixels[1:, 1:] & ~pixels[:-1, 1:] & ~pixels[1:, :-1]
    )
    down_left = (
        pixels[:-1, 1:] & pixels[1:, :-1] & ~pixels[:-1, :-1] & ~pixels[1:, 1:]
    )
    diagonal = np.count_nonzero(down_right) + np.count_nonzero(down_left)

    # NumPy 2 부터 count_nonzero가 np.int64를 돌려주므로 결과가 np.float64로
    # 승격됩니다. 비용 산출부의 positive()는 정확한 타입 일치를 요구하므로
    # 여기서 파이썬 float으로 되돌립니다.
    return float(
        horizontal * scale_x
        + vertical * scale_y
        + diagonal * float(np.hypot(scale_x, scale_y))
    )


def build_wall_mask(binary_mask: np.ndarray) -> np.ndarray:
    """균열을 둘러싼 벽면 링 마스크를 만듭니다 (팽창 - 원본)."""
    kernel = np.ones((S.wall_kernel_size, S.wall_kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(binary_mask, kernel, iterations=1)
    return cv2.subtract(dilated, binary_mask)


# ============================================================
# 1. 균열 치수 측정 (길이 / 폭 / 면적)
# ============================================================

def calculate_crack_metrics(
    binary_mask: np.ndarray,
    depth_image: np.ndarray,
    wall_mask: np.ndarray,
    fx: float,
    fy: float,
    marker_mm_per_pixel: float | None = None,
) -> dict[str, Any]:
    """ROI 크기의 마스크에서 균열의 길이·폭·면적을 mm 단위로 계산합니다.

    스케일은 ArUco 마커가 있으면 그것을, 없으면 주변 벽면의 Depth 중앙값을 씁니다.
    모든 배열은 ROI 좌표계이며, 반환되는 skeleton도 ROI 좌표계입니다.
    """
    empty = {
        "length_mm": 0.0,
        "max_width_mm": 0.0,
        "mean_width_mm": 0.0,
        "raw_max_width_mm": None,
        "widest_point_roi": None,
        "area_mm2": 0.0,
        "contour": None,
        "skeleton": np.zeros_like(binary_mask),
        "scale_source": "unavailable",
        "mm_per_pixel": None,
        "median_depth_mm": 0.0,
    }

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return empty

    largest = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(largest, 0.005 * cv2.arcLength(largest, True), True)

    # 스케일 결정
    median_depth = 0.0
    if marker_mm_per_pixel is None:
        wall_depths = depth_image[wall_mask > 0]
        valid = wall_depths[wall_depths > 0]

        if valid.size < 20:
            crack_depths = depth_image[binary_mask > 0]
            valid = crack_depths[crack_depths > 0]

        if valid.size == 0:
            empty["contour"] = approx
            return empty

        median_depth = float(np.median(valid))
        if median_depth <= 0:
            empty["contour"] = approx
            return empty

        scale_x = median_depth / float(fx)
        scale_y = median_depth / float(fy)
        scale_source = "depth_approximation"
    else:
        scale_x = scale_y = float(marker_mm_per_pixel)
        scale_source = "aruco"

    mm_per_pixel = float((scale_x + scale_y) / 2.0)

    # 중심선 · 길이
    skeleton = thin_binary_mask(binary_mask)
    length_mm = skeleton_length_mm(skeleton, scale_x, scale_y)

    # 폭: 중심선 위 거리변환 값의 2배. 최대값 대신 상위 95%로 이상치를 줄입니다.
    distance = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    ridge = distance[skeleton > 0]
    if ridge.size == 0:
        ridge = distance[binary_mask > 0]

    if ridge.size:
        max_width_px = float(np.percentile(ridge, 95)) * 2.0
        mean_width_px = float(np.median(ridge)) * 2.0
        candidates = np.where(skeleton > 0, distance, 0)
        if not np.any(candidates):
            candidates = distance
        peak_y, peak_x = np.unravel_index(int(np.argmax(candidates)), candidates.shape)
        raw_max_width_mm = float(candidates[peak_y, peak_x]) * 2.0 * mm_per_pixel
        widest_point = [int(peak_x), int(peak_y)]
    else:
        max_width_px = mean_width_px = 0.0
        raw_max_width_mm = None
        widest_point = None

    # 면적은 마스크 픽셀 수 기준 (여러 조각이 있어도 모두 반영)
    area_px = float(np.count_nonzero(binary_mask))

    return {
        "length_mm": length_mm,
        "max_width_mm": max_width_px * mm_per_pixel,
        "mean_width_mm": mean_width_px * mm_per_pixel,
        "raw_max_width_mm": raw_max_width_mm,
        "widest_point_roi": widest_point,
        "area_mm2": area_px * scale_x * scale_y,
        "contour": approx,
        "skeleton": skeleton,
        "scale_source": scale_source,
        "mm_per_pixel": mm_per_pixel,
        "median_depth_mm": median_depth,
    }


def highres_scale() -> float:
    """preview 픽셀 1개에 해당하는 고해상도 픽셀 수 (1920/640 = 3.0)."""
    return S.highres_width / float(S.width)


def refine_mask_highres(
    gray: np.ndarray,
    polygon: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """preview 마스크를 씨앗 삼아 고해상도에서 균열 경계를 다시 찾습니다.

    저해상도 마스크를 그냥 확대하면 경계 정밀도는 그대로이므로 아무 이득이
    없습니다. YOLO 마스크는 '어디를 볼지'로만 쓰고, 실제 경계는 고해상도
    그레이스케일에서 적응형 임계로 다시 뽑아야 폭 분해능이 실제로 올라갑니다.
    반환 좌표계는 고해상도 ROI 기준입니다.
    """
    scale = highres_scale()
    points = np.rint(polygon.astype(np.float32) * scale).astype(np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, S.highres_width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, S.highres_height - 1)

    margin = S.highres_margin_px + S.highres_seed_dilate_px
    x0 = max(0, int(points[:, 0].min()) - margin)
    y0 = max(0, int(points[:, 1].min()) - margin)
    x1 = min(S.highres_width, int(points[:, 0].max()) + margin + 1)
    y1 = min(S.highres_height, int(points[:, 1].max()) + margin + 1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    seed = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(seed, [points - np.array([x0, y0])], 255)
    seed_area = int(np.count_nonzero(seed))
    if seed_area == 0:
        return None

    roi = gray[y0:y1, x0:x1]
    roi = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(roi)

    block = S.highres_block_px | 1  # 적응형 임계는 홀수 블록만 받습니다.
    binary = cv2.adaptiveThreshold(
        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, S.highres_bias,
    )

    # 균열은 주변보다 어둡습니다. 다만 그림자·얼룩도 같이 잡히므로
    # YOLO가 지목한 범위를 조금 넓힌 곳 밖은 버립니다.
    kernel = np.ones((S.highres_seed_dilate_px, S.highres_seed_dilate_px), np.uint8)
    binary &= cv2.dilate(seed, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return None

    keep = [
        label for label in range(1, count)
        if np.any(seed[labels == label])
    ]
    if not keep:
        return None

    mask = np.isin(labels, keep).astype(np.uint8) * 255
    area = int(np.count_nonzero(mask))
    if area == 0 or area > seed_area * S.highres_area_guard:
        # 임계가 번져 벽면까지 삼킨 경우입니다. 저해상도 결과를 씁니다.
        return None

    return mask, (x0, y0)


def measure_highres(
    gray: np.ndarray,
    polygon: np.ndarray,
    mm_per_pixel_preview: float,
) -> dict[str, Any] | None:
    """고해상도에서 다시 뽑은 마스크로 치수를 계산합니다.

    축척은 preview 기준 mm/px를 고해상도 픽셀로 나눈 값입니다. ArUco를
    1080p에서 검출했으므로 이 환산은 정확합니다.
    """
    refined = refine_mask_highres(gray, polygon)
    if refined is None:
        return None

    mask, origin = refined
    scale = highres_scale()
    mm_per_pixel = mm_per_pixel_preview / scale

    metrics = calculate_crack_metrics(
        mask,
        np.zeros_like(mask, dtype=np.uint16),  # 축척을 직접 주므로 depth 불필요
        np.zeros_like(mask),
        1.0,
        1.0,
        mm_per_pixel,
    )
    if metrics["mm_per_pixel"] is None:
        return None

    widest = metrics["widest_point_roi"]
    metrics["widest_point_frame"] = (
        [(origin[0] + widest[0]) / scale, (origin[1] + widest[1]) / scale]
        if widest is not None
        else None
    )
    metrics["highres_mask_px"] = int(np.count_nonzero(mask))
    return metrics


# ============================================================
# 2. 깊이(단차) 측정
# ============================================================

def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PCA로 평면을 맞춰 (중심, 법선)을 반환합니다."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, eigenvectors = np.linalg.eigh(centered.T @ centered)
    normal = eigenvectors[:, 0]
    if normal[2] < 0:
        normal = -normal
    return centroid, normal


def calculate_crack_depth(
    binary_mask: np.ndarray,
    depth_image: np.ndarray,
    wall_mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    offset_x: int,
    offset_y: int,
) -> dict[str, Any]:
    """주변 벽면 평면 대비 균열 내부의 함몰 깊이를 추정합니다.

    binary_mask / wall_mask / depth_image는 ROI 좌표계이고,
    intrinsics는 전체 프레임 기준이므로 offset으로 보정합니다.
    반환되는 pixel_xs / pixel_ys는 전체 프레임 좌표계입니다.
    """
    base = {
        "depth_mm": None,
        "valid_ratio": 0.0,
        "recessed_ratio": 0.0,
        "noise_mm": None,
        "threshold_mm": None,
        "status": "빈 균열 마스크",
    }

    mask_pixels = int(np.count_nonzero(binary_mask))
    if mask_pixels == 0:
        return base

    crack_ys, crack_xs = np.nonzero(binary_mask)
    crack_depths = depth_image[crack_ys, crack_xs]
    crack_valid = crack_depths > 0
    base["valid_ratio"] = float(np.count_nonzero(crack_valid) / mask_pixels)

    wall_ys, wall_xs = np.nonzero(wall_mask)
    wall_depths = depth_image[wall_ys, wall_xs]
    wall_valid = wall_depths > 0
    wall_xs, wall_ys, wall_depths = (
        wall_xs[wall_valid],
        wall_ys[wall_valid],
        wall_depths[wall_valid],
    )

    if wall_depths.size < 20:
        base["status"] = "주변 벽면 Depth 부족"
        return base

    wall_points = deproject_pixels_to_points_mm(
        wall_xs + offset_x, wall_ys + offset_y, wall_depths, fx, fy, cx, cy
    )

    # 1차 평면 맞춤 후 이상치를 걸러 한 번 더 맞춥니다.
    centroid, normal = _fit_plane(wall_points)
    residuals = (wall_points - centroid) @ normal
    spread = 1.4826 * float(np.median(np.abs(residuals - np.median(residuals))))
    if spread > 0:
        inliers = np.abs(residuals - np.median(residuals)) < 2.5 * spread
        if np.count_nonzero(inliers) >= 20:
            centroid, normal = _fit_plane(wall_points[inliers])
            residuals = (wall_points - centroid) @ normal

    noise_mm = 1.4826 * float(
        np.median(np.abs(residuals - np.median(residuals)))
    )
    threshold_mm = max(S.min_depth_signal_mm, S.depth_noise_multiplier * noise_mm)

    base["noise_mm"] = noise_mm
    base["threshold_mm"] = threshold_mm

    crack_xs, crack_ys, crack_depths = (
        crack_xs[crack_valid],
        crack_ys[crack_valid],
        crack_depths[crack_valid],
    )

    if crack_depths.size < 5:
        base["status"] = "유효 Depth 픽셀 부족"
        return base

    crack_points = deproject_pixels_to_points_mm(
        crack_xs + offset_x, crack_ys + offset_y, crack_depths, fx, fy, cx, cy
    )
    signed_distances = (crack_points - centroid) @ normal

    recessed_ratio = float(
        np.count_nonzero(signed_distances > threshold_mm) / signed_distances.size
    )
    estimated_depth_mm = float(
        np.percentile(signed_distances, S.depth_percentile)
    )

    base["recessed_ratio"] = recessed_ratio
    base["pixel_xs"] = crack_xs + offset_x
    base["pixel_ys"] = crack_ys + offset_y
    base["signed_distances_mm"] = signed_distances

    if base["valid_ratio"] < S.min_depth_valid_ratio:
        base["status"] = "균열 내부 Depth 부족"
        return base

    if recessed_ratio < S.min_recessed_pixel_ratio or estimated_depth_mm <= threshold_mm:
        base["status"] = "벽면 대비 깊이 신호 부족"
        return base

    base["depth_mm"] = estimated_depth_mm
    base["status"] = "측정 가능"
    return base


# ============================================================
# 3. 위험도 판정
# ============================================================

def grade_crack(
    width_mm: float,
    mm_per_pixel: float | None,
    material: str = "rc",
    limit_px: float = 2.0,
) -> dict[str, Any]:
    """균열 폭으로 상태평가 등급(a~e)과 예비등급·권고 조치를 매깁니다.

    기준표는 부재 재질마다 다릅니다. 철근콘크리트는 0.1/0.3/0.5/1.0 mm,
    프리스트레스 콘크리트는 0.2/0.3/0.5 mm가 경계이고, 콘크리트 가로보에는
    e등급이 없습니다. 강재 부재는 균열폭 기준 자체가 없어 판정을 보류합니다.

    폭과 깊이는 서로 다른 지표이므로 등급은 폭으로만 결정하고,
    Depth 신뢰도와 측정 해상도는 별도 정보로 전달합니다.
    """
    pending = {
        "risk_code": "pending",
        "risk_level": "판정 보류",
        "grade": None,
        "preliminary_grade": "판정 보류",
        "action": "재촬영 또는 근접 확인",
        "measurement_limit_mm": None,
        "below_resolution": False,
        "grade_basis": None,
    }

    if mm_per_pixel is None or width_mm <= 0:
        pending["risk_level"] = "판정 보류 (치수 미산출)"
        return pending

    limit_mm = limit_px * mm_per_pixel  # 이 폭 이하는 판정 보류
    pending["measurement_limit_mm"] = round(limit_mm, 3)
    pending["below_resolution"] = width_mm <= limit_mm
    if pending["below_resolution"]:
        pending["risk_level"] = "판정 보류 (측정 한계 이하)"
        return pending

    table = CRACK_GRADE_TABLES.get(material)
    if table is None:
        pending["risk_level"] = f"판정 보류 ({MATERIAL_LABELS.get(material, material)})"
        return pending

    grade = GRADE_FALLBACK[material]
    for upper, label in table:
        if width_mm < upper:
            grade = label
            break

    preliminary, action = GRADE_ACTIONS[grade]
    risk_level = f"{grade}등급 · {preliminary}"
    return {
        "risk_code": GRADE_RISK_CODES[grade],
        "risk_level": risk_level,
        "grade": grade,
        "preliminary_grade": preliminary,
        "action": action,
        "measurement_limit_mm": round(limit_mm, 3),
        "below_resolution": pending["below_resolution"],
        "grade_basis": f"{MATERIAL_LABELS.get(material, material)} 균열폭 기준",
    }


# ============================================================
# 점검 기본정보 (보고서 머리말)
# ============================================================

def load_inspection_config() -> dict[str, Any]:
    global inspection

    path = S.inspection_config_path
    if not path.exists():
        return inspection

    try:
        with path.open("r", encoding="utf-8") as file:
            saved = json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        print(f"[설정] inspection_config.json 읽기 실패: {error}")
        return inspection

    with inspection_lock:
        inspection = {**DEFAULT_INSPECTION, **saved}
        return dict(inspection)


def save_inspection_config(values: dict[str, Any]) -> dict[str, Any]:
    global inspection

    with inspection_lock:
        merged = {**inspection}
        for key in DEFAULT_INSPECTION:
            if key in values:
                merged[key] = values[key]
        inspection = merged
        snapshot = dict(merged)

    with S.inspection_config_path.open("w", encoding="utf-8") as file:
        json.dump(snapshot, file, ensure_ascii=False, indent=2)

    return snapshot


def current_inspection() -> dict[str, Any]:
    with inspection_lock:
        return dict(inspection)


# ============================================================
# 저장 / 이력
# ============================================================

def save_capture(image: np.ndarray, measurements: dict[str, Any]) -> tuple[Path, Path]:
    """현재 합성 화면과 같은 시점의 측정값을 PNG와 JSON으로 저장합니다."""
    S.capture_dir.mkdir(parents=True, exist_ok=True)

    captured_at = datetime.now()
    stem = captured_at.strftime("crack_%Y%m%d_%H%M%S_%f")[:-3]
    image_path = S.capture_dir / f"{stem}.png"
    json_path = S.capture_dir / f"{stem}.json"

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG 인코딩에 실패했습니다.")
    encoded.tofile(str(image_path))

    metadata = {
        "captured_at": captured_at.isoformat(timespec="milliseconds"),
        "image_file": image_path.name,
        "inspection": current_inspection(),
        "measurements": measurements,
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    return image_path, json_path


def append_history(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """동일 관측은 기존 손상 번호로 갱신하고 SQLite에 보관합니다."""
    with history_lock:
        return get_observation_store().save(records)


# ============================================================
# DepthAI 파이프라인
# ============================================================

def create_pipeline() -> dai.Pipeline:
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam_rgb.setResolution(
        dai.ColorCameraProperties.SensorResolution.THE_1080_P
    )
    # S.width:S.height가 센서 종횡비(16:9)와 같아야 크롭 없이 전체 FOV가 나옵니다.
    cam_rgb.setPreviewKeepAspectRatio(True)
    cam_rgb.setPreviewSize(S.width, S.height)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(S.fps)

    left = pipeline.create(dai.node.MonoCamera)
    left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    left.setFps(S.fps)

    right = pipeline.create(dai.node.MonoCamera)
    right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    right.setFps(S.fps)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    # CAM_A 정렬 depth는 컬러 센서의 전체 FOV를 덮으므로, 출력 크기를 preview와
    # 같은 16:9로 맞추면 두 영상이 화소 단위로 대응합니다.
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(S.width, S.height)

    # 진단 목적으로 후처리 필터는 모두 끕니다 (hole-filling이 균열을 메움).
    config = stereo.initialConfig.get()
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

    if S.use_highres_width:
        # preview와 같은 ISP 프레임의 고해상도 판본입니다. 시퀀스 번호가
        # 같으므로 어느 preview 프레임과 짝인지 정확히 알 수 있습니다.
        cam_rgb.setVideoSize(S.highres_width, S.highres_height)
        xout_video = pipeline.create(dai.node.XLinkOut)
        xout_video.setStreamName("video")
        cam_rgb.video.link(xout_video.input)

    return pipeline


# ============================================================
# 프레임 처리
# ============================================================

@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class FrameOverlays:
    """한 프레임에서 모은 시각화용 데이터."""
    crack_roi: np.ndarray
    depth_valid: np.ndarray
    depth_values: np.ndarray
    contours: list[np.ndarray] = field(default_factory=list)


def analyse_crack(
    polygon: np.ndarray,
    depth_image: np.ndarray,
    intrinsics: Intrinsics,
    marker_mm_per_pixel: float | None,
    highres_gray: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """마스크 하나를 ROI로 잘라 치수와 깊이를 계산합니다.

    전체 프레임(640×480)이 아니라 균열 주변만 처리하기 때문에
    세선화·거리변환·팽창 비용이 마스크 크기에 비례합니다.
    """
    points = polygon.astype(np.int32)
    if len(points) < 3:
        return None

    points[:, 0] = np.clip(points[:, 0], 0, S.width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, S.height - 1)

    margin = S.roi_margin_px + S.wall_kernel_size
    x0 = max(0, int(points[:, 0].min()) - margin)
    y0 = max(0, int(points[:, 1].min()) - margin)
    x1 = min(S.width, int(points[:, 0].max()) + margin + 1)
    y1 = min(S.height, int(points[:, 1].max()) + margin + 1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None

    roi_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [points - np.array([x0, y0])], 255)

    if np.count_nonzero(roi_mask) < S.min_mask_pixels:
        return None

    roi_depth = depth_image[y0:y1, x0:x1]
    wall_mask = build_wall_mask(roi_mask)  # 치수·깊이 계산이 함께 사용

    metrics = calculate_crack_metrics(
        roi_mask,
        roi_depth,
        wall_mask,
        intrinsics.fx,
        intrinsics.fy,
        marker_mm_per_pixel,
    )
    depth = calculate_crack_depth(
        roi_mask,
        roi_depth,
        wall_mask,
        intrinsics.fx,
        intrinsics.fy,
        intrinsics.cx,
        intrinsics.cy,
        x0,
        y0,
    )

    # 폭·길이는 고해상도 재측정 결과가 있으면 그것으로 대체합니다.
    # 깊이는 depth 맵이 preview 해상도라 저해상도 결과를 그대로 씁니다.
    if highres_gray is not None and metrics["mm_per_pixel"] is not None:
        highres = measure_highres(highres_gray, points, metrics["mm_per_pixel"])
        if highres is not None:
            frame_point = highres.pop("widest_point_frame", None)
            highres["widest_point_roi"] = (
                [int(round(frame_point[0])) - x0, int(round(frame_point[1])) - y0]
                if frame_point is not None
                else None
            )
            highres["scale_source"] = metrics["scale_source"] + "_highres"
            highres["median_depth_mm"] = metrics["median_depth_mm"]
            highres["skeleton"] = metrics["skeleton"]   # 표시는 preview 좌표계
            highres["contour"] = metrics["contour"]
            metrics = highres

    return {
        "points": points,
        "origin": (x0, y0),
        "mask": roi_mask,
        "metrics": metrics,
        "depth": depth,
    }


def build_report(
    crack_id: str,
    confidence: float,
    analysis: dict[str, Any],
    cost_config: dict[str, Any],
    context: dict[str, Any],
    setup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = analysis["metrics"]
    depth = analysis["depth"]

    risk = grade_crack(
        metrics["max_width_mm"],
        metrics["mm_per_pixel"],
        context.get("member_material", "rc"),
        limit_px=(
            S.highres_limit_px
            if str(metrics["scale_source"]).endswith("_highres")
            else 2.0
        ),
    )
    repair = estimate_repair_cost(
        metrics["length_mm"], metrics["max_width_mm"], cost_config, context,
        measurement_valid=risk["grade"] is not None and not risk["below_resolution"],
    )
    if setup is not None:
        repair = estimate_depth_scenarios(
            metrics["length_mm"], metrics["mean_width_mm"], setup,
            length_valid=metrics["mm_per_pixel"] is not None,
            width_valid=(metrics["mm_per_pixel"] is not None
                         and metrics["mean_width_mm"] > 2 * metrics["mm_per_pixel"]),
        )
    depth_mm = depth["depth_mm"]

    return {
        "id": crack_id,
        "bbox": [int(analysis["points"][:, 0].min()),
                 int(analysis["points"][:, 1].min()),
                 int(analysis["points"][:, 0].max()) + 1,
                 int(analysis["points"][:, 1].max()) + 1],
        "inspection_key": json.dumps(
            {key: context.get(key) for key in (
                "bridge_id", "bridge_name", "site", "member_group", "member",
                "member_label", "member_material", "member_position"
            )}, sort_keys=True, ensure_ascii=False,
        ),
        "damage_no": None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "defect": "균열",
        "confidence": round(confidence, 3),

        "member_group": context.get("member_group"),
        "member": context.get("member_label") or context.get("member"),
        "member_material": context.get("member_material"),
        "member_material_label": MATERIAL_LABELS.get(
            context.get("member_material", ""), "-"
        ),
        "position": context.get("member_position"),

        "length_mm": round(float(metrics["length_mm"]), 1),
        "length_method": "skeleton_total_length",
        "max_width_mm": round(float(metrics["max_width_mm"]), 2),
        "width_p95_mm": round(float(metrics["max_width_mm"]), 4),
        "raw_max_width_mm": round(metrics["raw_max_width_mm"], 4) if metrics["raw_max_width_mm"] is not None else None,
        "widest_point_px": [int(metrics["widest_point_roi"][i] + analysis["origin"][i]) for i in (0, 1)] if metrics["widest_point_roi"] is not None else None,
        "width_profile_version": WIDTH_VERSION,
        "mean_width_mm": round(float(metrics["mean_width_mm"]), 2),
        "area_mm2": round(float(metrics["area_mm2"]), 1),

        "dimension_scale_source": metrics["scale_source"],
        "mm_per_pixel": (
            round(metrics["mm_per_pixel"], 4)
            if metrics["mm_per_pixel"] is not None
            else None
        ),
        "aruco_marker_size_mm": (
            S.aruco_marker_size_mm
            if metrics["scale_source"] == "aruco"
            else None
        ),
        "measurement_limit_mm": risk["measurement_limit_mm"],
        "below_resolution": risk["below_resolution"],
        "width_limit_px": (
            S.highres_limit_px
            if str(metrics["scale_source"]).endswith("_highres")
            else 2.0
        ),

        "recommended_repair_method": repair["method_label"],
        "repair_quantity_m": round(repair["length_m"], 3),
        "unit_price_krw_per_m": repair["unit_price_krw_per_m"],
        "estimated_repair_cost_krw": (
            round(repair["estimated_cost_krw"])
            if repair["estimated_cost_krw"] is not None
            else None
        ),
        "cost_status": repair["status"],
        "cost_model_version": repair["model_version"],
        "cost_scenarios": repair.get("scenarios", []),
        "cost_basis": repair["basis"],
        "cost_breakdown": repair["breakdown"],
        "cost_exclusions": repair["exclusions"],

        "depth_diff_mm": round(float(depth_mm), 1) if depth_mm is not None else None,
        "depth_reliable": depth_mm is not None,
        "depth_valid_ratio": round(depth["valid_ratio"], 3),
        "depth_recessed_ratio": round(depth["recessed_ratio"], 3),
        "depth_noise_mm": (
            round(depth["noise_mm"], 2) if depth["noise_mm"] is not None else None
        ),
        "depth_threshold_mm": (
            round(depth["threshold_mm"], 2)
            if depth["threshold_mm"] is not None
            else None
        ),
        "depth_status": depth["status"],

        "grade": risk["grade"],
        "preliminary_grade": risk["preliminary_grade"],
        "action": risk["action"],
        "grade_basis": (risk["grade_basis"] + " · P95 기반 프로젝트 예비 분류, 구조 안전등급 아님") if risk["grade_basis"] else None,
        "risk_level": risk["risk_level"],
        "risk_code": risk["risk_code"],
        "image_url": None,
    }


RISK_COLORS = {
    "danger": (0, 0, 255),
    "caution": (0, 165, 255),
    "normal": (255, 0, 0),
    "pending": (0, 255, 255),
}


def draw_crack(
    canvas: np.ndarray,
    analysis: dict[str, Any],
    report: dict[str, Any],
) -> None:
    x0, y0 = analysis["origin"]
    metrics = analysis["metrics"]
    color = RISK_COLORS.get(report["risk_code"], (0, 255, 255))

    contour = metrics["contour"]
    if contour is not None:
        cv2.drawContours(canvas, [contour + np.array([x0, y0])], -1, color, 2)

    skeleton = metrics["skeleton"]
    region = canvas[y0:y0 + skeleton.shape[0], x0:x0 + skeleton.shape[1]]
    region[skeleton > 0] = (0, 255, 0)
    if report.get("widest_point_px") is not None:
        center = tuple(report["widest_point_px"])
        cv2.circle(canvas, center, 6, (255, 0, 255), 2)
        cv2.putText(canvas, "WMAX", (center[0] + 8, max(12, center[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

    points = analysis["points"]
    x_text = max(5, int(points[:, 0].min()))
    y_text = min(S.height - 80, int(points[:, 1].max()) + 20)

    lines = [
        (f"L: {report['length_mm']:.1f}mm", (0, 255, 255)),
        (f"W95: {report['max_width_mm']:.2f}mm", color),
        (f"A: {report['area_mm2']:.0f}mm2", (0, 255, 255)),
    ]

    depth_text = (
        f"D: {report['depth_diff_mm']:.1f}mm"
        if report["depth_diff_mm"] is not None
        else "D: N/A"
    )
    depth_text += (
        f" V:{report['depth_valid_ratio']:.0%} S:{report['depth_recessed_ratio']:.0%}"
    )
    lines.append(
        (depth_text, (0, 165, 255) if report["depth_reliable"] else (160, 160, 160))
    )

    for index, (text, text_color) in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (x_text, y_text + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            2,
        )


def render_depth_panel(overlays: FrameOverlays) -> np.ndarray:
    normalized = np.clip(
        overlays.depth_values / S.local_depth_vis_max_mm * 255.0, 0, 255
    ).astype(np.uint8)

    panel = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    panel[~overlays.crack_roi] = 0
    panel[overlays.crack_roi & ~overlays.depth_valid] = (80, 80, 80)

    for contour in overlays.contours:
        cv2.polylines(panel, [contour], True, (255, 255, 255), 1)

    cv2.putText(
        panel,
        f"LOCAL CRACK DEPTH 0-{S.local_depth_vis_max_mm:.0f} mm",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        panel,
        "BLUE=surface  RED=deep  GRAY=invalid",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    return panel


# ============================================================
# 카메라 메인 루프
# ============================================================

def camera_loop() -> None:
    """장치 오류가 나면 잠시 쉬었다가 다시 연결을 시도합니다."""
    backoff = 2.0

    while not stop_event.is_set():
        try:
            run_camera_session()
            backoff = 2.0
        except Exception as error:  # 장치 분리, USB 오류 등
            print(f"\n[OAK-D 오류] {error}")
            set_status(camera_status="error", error=str(error))

        if stop_event.is_set():
            break

        print(f"{backoff:.0f}초 후 카메라 재연결을 시도합니다.")
        stop_event.wait(backoff)
        backoff = min(backoff * 1.5, 15.0)

    if S.show_local_window:
        cv2.destroyAllWindows()


def run_camera_session() -> None:
    assert model is not None, "YOLO 모델이 로드되지 않았습니다."

    pipeline = create_pipeline()
    aruco = ArucoScaleTracker()

    frame_count = 0
    smoothed_fps = 0.0
    last_history_at = 0.0
    last_auto_capture_at = 0.0
    pending_capture: list[dict[str, Any]] = []
    tracker = ObservationTracker()

    with dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER) as device:
        usb_speed = str(device.getUsbSpeed())
        print(f"USB 연결 속도: {usb_speed}")
        set_status(usb_speed=usb_speed, camera_status="online", error=None)

        q_rgb = device.getOutputQueue("rgb", maxSize=2, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=2, blocking=False)
        q_video = (
            device.getOutputQueue("video", maxSize=2, blocking=False)
            if S.use_highres_width
            else None
        )

        # keepAspectRatio 기본값(True)은 종횡비가 다를 때만 크롭 보정을 넣습니다.
        # S.width:S.height를 센서와 같은 16:9로 두었으므로 여기서는 순수 스케일입니다.
        matrix = device.readCalibration().getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A, S.width, S.height
        )
        intrinsics = Intrinsics(
            fx=matrix[0][0], fy=matrix[1][1], cx=matrix[0][2], cy=matrix[1][2]
        )
        print(
            f"Intrinsics: fx={intrinsics.fx:.3f}, fy={intrinsics.fy:.3f}, "
            f"cx={intrinsics.cx:.3f}, cy={intrinsics.cy:.3f}"
        )

        while not stop_event.is_set():
            started_at = time.perf_counter()

            rgb_packet = q_rgb.get()
            depth_packet = q_depth.get()
            frame_count += 1

            color_image = rgb_packet.getCvFrame()
            depth_image = depth_packet.getFrame()

            # preview와 같은 ISP 프레임의 1080p 판본을 시퀀스 번호로 맞춥니다.
            # 어긋나면 그 프레임만 저해상도로 떨어뜨립니다 (움직이는 중에
            # 엉뚱한 위치를 재는 것보다 낫습니다).
            highres_gray = None
            if q_video is not None:
                target = rgb_packet.getSequenceNum()
                video_packet = None
                while True:
                    packet = q_video.tryGet()
                    if packet is None:
                        break
                    video_packet = packet
                    if packet.getSequenceNum() >= target:
                        break
                if video_packet is not None and abs(
                    video_packet.getSequenceNum() - target
                ) <= S.highres_max_seq_lag:
                    highres_gray = cv2.cvtColor(
                        video_packet.getCvFrame(), cv2.COLOR_BGR2GRAY
                    )

            if color_image.shape[:2] != (S.height, S.width):
                color_image = cv2.resize(color_image, (S.width, S.height))
            if depth_image.shape[:2] != (S.height, S.width):
                depth_image = cv2.resize(
                    depth_image, (S.width, S.height), interpolation=cv2.INTER_NEAREST
                )

            if frame_count == 1 or frame_count % S.aruco_detect_interval == 0:
                # 고해상도에서 검출하면 마커가 3배 커져 코너 오차가 1/3로 줄고,
                # mm/px는 preview 좌표계로 환산되어 돌아옵니다.
                if highres_gray is not None:
                    aruco.update(highres_gray, S.width / float(S.highres_width))
                else:
                    aruco.update(color_image)

            if frame_count % 60 == 0:
                load_repair_cost_config()

            inference_started_at = time.perf_counter()
            result = model(
                color_image,
                imgsz=S.yolo_imgsz,
                conf=S.conf_threshold,
                verbose=False,
            )[0]
            inference_ms = (time.perf_counter() - inference_started_at) * 1000.0

            annotated = result.plot()
            overlays = FrameOverlays(
                crack_roi=np.zeros((S.height, S.width), dtype=bool),
                depth_valid=np.zeros((S.height, S.width), dtype=bool),
                depth_values=np.zeros((S.height, S.width), dtype=np.float32),
            )
            measurements: list[dict[str, Any]] = []
            context = current_inspection()
            try:
                setup = estimate_setup_store.snapshot()
            except (OSError, ValueError):
                setup = {"configured": False}

            if result.masks is not None and result.boxes is not None:
                for index, polygon in enumerate(result.masks.xy):
                    class_id = int(result.boxes.cls[index].item())
                    if result.names[class_id].lower() != "crack":
                        continue

                    analysis = analyse_crack(
                        polygon, depth_image, intrinsics, aruco.mm_per_pixel,
                        highres_gray,
                    )
                    if analysis is None:
                        continue

                    report = build_report(
                        f"CRK-{frame_count:06d}-{index + 1:02d}",
                        float(result.boxes.conf[index].item()),
                        analysis,
                        repair_cost_config,
                        context,
                        setup,
                    )
                    measurements.append(report)
                    report["aruco_marker_id"] = aruco.marker_id
                    report["scale_stale"] = aruco.is_stale

                    x0, y0 = analysis["origin"]
                    mask = analysis["mask"]
                    overlays.crack_roi[
                        y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]
                    ] |= mask > 0
                    overlays.contours.append(analysis["points"])

                    depth = analysis["depth"]
                    xs = depth.get("pixel_xs")
                    ys = depth.get("pixel_ys")
                    signed = depth.get("signed_distances_mm")
                    if xs is not None and ys is not None and signed is not None:
                        clipped = np.clip(signed, 0.0, S.local_depth_vis_max_mm)
                        overlays.depth_values[ys, xs] = np.maximum(
                            overlays.depth_values[ys, xs], clipped
                        )
                        overlays.depth_valid[ys, xs] = True

                    draw_crack(annotated, analysis, report)

            danger_present = any(
                report["risk_code"] == "danger" for report in measurements
            )
            now = time.monotonic()
            tracker.observe(measurements, now)
            if measurements and (
                now - last_history_at >= S.history_min_interval_s
                or (danger_present and now - last_history_at >= 1.0)
            ):
                logged = append_history(measurements)
                last_history_at = now

                # 보고서의 '손상별 상세 페이지'에 붙일 이미지는 합성 화면이
                # 만들어진 뒤에 저장합니다.
                if now - last_auto_capture_at >= S.auto_capture_cooldown_s:
                    pending_capture = [
                        record for record in logged
                        if record["grade"]
                        and GRADE_ORDER.index(record["grade"])
                        >= GRADE_ORDER.index(S.auto_capture_min_grade)
                    ]

            cv2.putText(
                annotated,
                f"Cracks: {len(measurements)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            if aruco.mm_per_pixel is not None:
                scale_text = (
                    f"ArUco {aruco.marker_id}: {aruco.mm_per_pixel:.3f} mm/px"
                    + (" (hold)" if aruco.is_stale else "")
                )
                scale_color = (0, 255, 0) if not aruco.is_stale else (0, 200, 200)
                if aruco.corners is not None:
                    corner_points = aruco.corners.astype(np.int32)
                    cv2.polylines(annotated, [corner_points], True, (0, 255, 0), 2)
            else:
                scale_text = "Scale: depth approximation (place 50mm ArUco)"
                scale_color = (0, 165, 255)

            cv2.putText(
                annotated,
                scale_text,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                scale_color,
                2,
            )

            combined = np.hstack((annotated, render_depth_panel(overlays)))

            if pending_capture:
                try:
                    image_path, _ = save_capture(
                        combined,
                        {
                            "crack_count": len(measurements),
                            "detections": pending_capture,
                            "trigger": "auto",
                        },
                    )
                    for record in pending_capture:
                        record["image_url"] = f"/captures/{image_path.name}"
                    with history_lock:
                        get_observation_store().attach_image(
                            pending_capture, f"/captures/{image_path.name}"
                        )
                    last_auto_capture_at = time.monotonic()
                except (OSError, RuntimeError) as error:
                    print(f"[캡처] 자동 저장 실패: {error}")
                pending_capture = []

            elapsed = time.perf_counter() - started_at
            current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            smoothed_fps = (
                current_fps
                if smoothed_fps == 0.0
                else 0.9 * smoothed_fps + 0.1 * current_fps
            )

            cv2.putText(
                combined,
                f"FPS: {smoothed_fps:.1f} | Inference: {inference_ms:.0f} ms",
                (10, S.height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            broker.publish(combined)

            # 폭을 고해상도에서 재므로 측정 한계도 그 축척으로 알립니다.
            limit_scale = highres_scale() if S.use_highres_width else 1.0
            limit_px = S.highres_limit_px if S.use_highres_width else 2.0
            measurement_limit = (
                round(limit_px * aruco.mm_per_pixel / limit_scale, 3)
                if aruco.mm_per_pixel is not None
                else None
            )
            with data_lock:
                latest_data.update(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "crack_count": len(measurements),
                        "detections": measurements,
                        "fps": round(smoothed_fps, 1),
                        "inference_ms": round(inference_ms, 1),
                        "usb_speed": usb_speed,
                        "aruco_detected": aruco.mm_per_pixel is not None,
                        "aruco_stale": aruco.is_stale,
                        "aruco_marker_id": aruco.marker_id,
                        "aruco_scale_mm_per_pixel": (
                            round(aruco.mm_per_pixel, 4)
                            if aruco.mm_per_pixel is not None
                            else None
                        ),
                        "measurement_limit_mm": measurement_limit,
                        "camera_status": "online",
                        "error": None,
                    }
                )

            if S.show_local_window:
                cv2.imshow("OAK-D Lite Crack Detection", combined)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_event.set()
                    break


# ============================================================
# Flask 라우트
# ============================================================

def generate_frames():
    last_seq = 0
    while not stop_event.is_set():
        frame_bytes, last_seq = broker.jpeg(last_seq, timeout=1.0)
        if frame_bytes is None:
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.after_request
def add_no_cache_headers(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "dashboard.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/imu")
def api_imu():
    return jsonify(read_imu_state())


@app.route("/api/latest")
def api_latest():
    with data_lock:
        return jsonify(dict(latest_data))


@app.route("/api/history")
def api_history():
    limit = max(1, min(request.args.get("limit", type=int) or 200, 5000))
    with history_lock:
        records = [present_record(row) for row in get_observation_store().records(limit)]
    return jsonify({"count": len(records), "records": records})


@app.route("/api/history", methods=["DELETE"])
def api_history_clear():
    with history_lock:
        get_observation_store().clear()
    return jsonify({"success": True})


@app.route("/api/inspection")
def api_inspection():
    """보고서 머리말에 들어가는 점검 기본정보와 선택 가능한 부재 목록."""
    return jsonify({
        "inspection": current_inspection(),
        "member_groups": MEMBER_GROUPS,
        "materials": MATERIAL_LABELS,
    })


@app.route("/api/estimate-setup", methods=["GET", "POST"])
def api_estimate_setup():
    try:
        if request.method == "POST":
            payload = request.get_json(silent=True)
            if not isinstance(payload, dict):
                raise ValueError("입력값 형식이 올바르지 않습니다.")
            state = estimate_setup_store.save(payload.get("inputs"))
        else:
            state = estimate_setup_store.snapshot()
        return jsonify({"success": True, **state, "defaults": DEFAULT_INPUTS, "norms": METHODS,
                        "depth_scenarios_cm": DEPTH_SCENARIOS_CM})
    except ValueError as error:
        return jsonify({"success": False, "message": str(error)}), 400
    except OSError:
        return jsonify({"success": False, "message": "추정 설정 파일을 읽거나 저장하지 못했습니다."}), 500


@app.route("/api/inspection", methods=["POST"])
def api_inspection_save():
    payload = request.get_json(silent=True) or {}
    try:
        saved = save_inspection_config(payload)
    except OSError as error:
        return jsonify({"success": False, "message": str(error)}), 500
    return jsonify({"success": True, "inspection": saved})


@app.route("/api/report")
def api_report():
    """손상 종합표에 필요한 것을 한 번에 내려줍니다."""
    with history_lock:
        records = [present_record(row) for row in get_observation_store().records()]

    by_member: dict[str, int] = {}
    by_grade: dict[str, int] = {}

    for record in records:
        member = record.get("member") or "미지정"
        by_member[member] = by_member.get(member, 0) + 1
        grade = record.get("grade") or "판정 보류"
        by_grade[grade] = by_grade.get(grade, 0) + 1

    return jsonify({
        "inspection": current_inspection(),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total": len(records),
            "by_member": by_member,
            "by_grade": by_grade,
            **summarize_costs(records),
        },
        "records": records,
    })


@app.route("/api/captures")
def api_captures():
    if not S.capture_dir.exists():
        return jsonify({"count": 0, "captures": []})

    captures = []
    for json_path in sorted(S.capture_dir.glob("crack_*.json"), reverse=True)[:100]:
        image_path = json_path.with_suffix(".png")
        entry = {
            "name": json_path.stem,
            "json_url": f"/captures/{json_path.name}",
            "image_url": f"/captures/{image_path.name}" if image_path.exists() else None,
            "captured_at": None,
            "crack_count": None,
        }
        try:
            with json_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            entry["captured_at"] = payload.get("captured_at")
            entry["crack_count"] = payload.get("measurements", {}).get("crack_count")
        except (json.JSONDecodeError, OSError):
            pass
        captures.append(entry)

    return jsonify({"count": len(captures), "captures": captures})


@app.route("/captures/<path:filename>")
def serve_capture(filename: str):
    return send_from_directory(S.capture_dir, filename)


@app.route("/api/capture", methods=["POST"])
def api_capture():
    frame = broker.snapshot()
    if frame is None:
        return jsonify({
            "success": False,
            "message": "아직 영상이 들어오지 않았습니다. 카메라 상태를 확인하세요.",
        }), 503

    with data_lock:
        measurements = dict(latest_data)

    try:
        image_path, json_path = save_capture(frame, measurements)
    except (OSError, RuntimeError) as error:
        return jsonify({"success": False, "message": str(error)}), 500

    return jsonify({
        "success": True,
        "name": image_path.stem,
        "image": str(image_path),
        "json": str(json_path),
        "image_url": f"/captures/{image_path.name}",
        "json_url": f"/captures/{json_path.name}",
    })


@app.route("/healthz")
def healthz():
    with data_lock:
        status = latest_data["camera_status"]
    return jsonify({"status": status}), (200 if status == "online" else 503)


# ============================================================
# 실행
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="터널 균열 실시간 모니터링")
    parser.add_argument("--model", default=S.model_path, help="YOLO 가중치 경로")
    parser.add_argument("--host", default=S.host)
    parser.add_argument("--port", type=int, default=S.port)
    parser.add_argument("--conf", type=float, default=S.conf_threshold)
    parser.add_argument("--imgsz", type=int, default=S.yolo_imgsz)
    parser.add_argument("--threads", type=int, default=S.torch_threads)
    parser.add_argument(
        "--no-window", action="store_true", help="로컬 OpenCV 창을 띄우지 않습니다"
    )
    return parser.parse_args()


def main() -> None:
    global model

    args = parse_args()
    S.model_path = args.model
    S.host = args.host
    S.port = args.port
    S.conf_threshold = args.conf
    S.yolo_imgsz = args.imgsz
    S.torch_threads = args.threads
    S.show_local_window = not args.no_window

    torch.set_num_threads(S.torch_threads)
    print("YOLO 모델 로딩 중...")
    model = YOLO(S.model_path)
    print(f"YOLO 모델 로드 완료 (PyTorch CPU 스레드: {torch.get_num_threads()})")

    if not _HAS_XIMGPROC:
        print(
            "안내: opencv-contrib-python이 없어 세선화를 NumPy로 처리합니다. "
            "`pip install opencv-contrib-python`을 설치하면 더 빨라집니다."
        )

    load_repair_cost_config(force=True)
    load_inspection_config()
    with history_lock:
        get_observation_store()

    camera_thread = threading.Thread(target=camera_loop, daemon=True)
    camera_thread.start()

    print()
    print("=" * 46)
    print("웹 대시보드")
    print(f"http://127.0.0.1:{S.port}")
    print(f"다른 PC: http://<이 PC의 IP>:{S.port}")
    print("=" * 46)
    print()

    try:
        app.run(
            host=S.host,
            port=S.port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    finally:
        stop_event.set()
        camera_thread.join(timeout=3.0)


if __name__ == "__main__":
    main()
