import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# 1. 학습이 완료된 모델 로드 (가중치 경로 확인 필요)
model = YOLO("runs/segment/crack_seg_official/weights/best.pt")

# 2. RealSense 파이프라인 및 설정 초기화
pipeline = rs.pipeline()
config = rs.config()

# RGB와 Depth 스트림 활성화 (해상도 640x480, 30프레임)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 파이프라인 시작
profile = pipeline.start(config)

# 3. 깊이(Depth) 맵을 RGB 이미지 시점에 맞추기 위한 Align 객체 생성
# (Depth 렌즈와 RGB 렌즈의 물리적 위치 차이를 소프트웨어로 교정)
align_to = rs.stream.color
align = rs.align(align_to)

print("RealSense 카메라 구동을 시작합니다. (종료: 'q')")

try:
    while True:
        # 4. 카메라로부터 프레임 세트 읽기
        frames = pipeline.wait_for_frames()
        
        # 5. RGB 시점(Viewpoint)에 맞게 Depth 프레임 정렬
        aligned_frames = align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame() # (현재 코드에선 쓰지 않지만 뎁스맵이 준비됨)
        
        if not color_frame or not depth_frame:
            continue

        # RealSense 프레임을 OpenCV에서 쓸 수 있는 Numpy 배열로 변환
        color_image = np.asanyarray(color_frame.get_data())

        # 6. YOLO 모델로 실시간 추론 (GPU 사용)
        results = model(color_image, stream=True, device=0)

        # 7. 결과 시각화 및 화면 출력
        for result in results:
            annotated_frame = result.plot()
            cv2.imshow("RealSense Crack Segmentation", annotated_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 8. 카메라 장치 안전하게 해제
    pipeline.stop()
    cv2.destroyAllWindows()