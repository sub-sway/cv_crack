import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# 1. 학습된 모델 로드
model = YOLO("runs/segment/hanium_dacl10k_seg-5/weights/best.pt")

# 2. RealSense 파이프라인 및 스트림 설정
pipeline = rs.pipeline()
config = rs.config()

# RGB 및 Depth 스트림 활성화 (640x480 해상도, 30FPS)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 파이프라인 시작
profile = pipeline.start(config)

# ★ 뎁스 시점을 RGB 카메라 시점(Color Stream)으로 완벽하게 투영(Align)
align_to = rs.stream.color
align = rs.align(align_to)

print("Intel RealSense 카메라 스트리밍 시작...")

try:
    while True:
        # 프레임 세트 수신
        frames = pipeline.wait_for_frames()
        
        # 정렬(Alignment) 실행
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # RealSense 프레임을 Numpy 배열로 변환
        depth_image = np.asanyarray(aligned_depth_frame.get_data()) # 단위: 밀리미터(mm)
        color_image = np.asanyarray(color_frame.get_data())

        # YOLO 세그멘테이션 추론
        results = model(color_image, verbose=False)
        annotated_frame = results[0].plot()

        # 마스크가 탐지된 경우
        if results[0].masks is not None:
            for idx, polygon in enumerate(results[0].masks.xy):
                pts = np.array(polygon, np.int32)
                
                # 마스크 생성 및 영역 분리
                binary_mask = np.zeros_like(depth_image, dtype=np.uint8)
                cv2.fillPoly(binary_mask, [pts], 255)

                # 균열 내부 픽셀의 깊이 데이터만 추출
                crack_depth_values = depth_image[binary_mask == 255]
                valid_depths = crack_depth_values[crack_depth_values > 0]

                if len(valid_depths) > 0:
                    avg_depth = np.mean(valid_depths)
                    max_depth = np.max(valid_depths)
                    
                    # [수정된 부분] y_min 대신 y_max(바운딩 박스의 가장 아래쪽 좌표)를 구함
                    x_min = np.min(pts[:, 0])
                    y_max = np.max(pts[:, 1])
                    cv2.putText(annotated_frame, f"D_Avg:{avg_depth:.1f}mm", (x_min, y_max + 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(annotated_frame, f"D_Max:{max_depth:.1f}mm", (x_min, y_max + 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 뎁스 맵도 컬러로 변환하여 함께 출력 (시각적 디버깅용)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        
        # 화면 가로로 붙여서 출력 (좌: YOLO 결과, 우: 뎁스 맵)
        images = np.hstack((annotated_frame, depth_colormap))
        cv2.imshow('RealSense Crack Detection', images)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()