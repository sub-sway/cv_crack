import pyrealsense2 as rs
import numpy as np
import cv2
import math
from ultralytics import YOLO

def sort_skeleton_points(pts_array):
    if len(pts_array) == 0:
        return []
    
    pts_list = pts_array.tolist()
    ordered = [pts_list.pop(0)]
    
    while len(pts_list) > 0:
        last_pt = np.array(ordered[-1])
        pts_arr = np.array(pts_list)
        
        dist_sq = np.sum((pts_arr - last_pt)**2, axis=1)
        closest_idx = np.argmin(dist_sq)
        
        ordered.append(pts_list.pop(closest_idx))
        
    return np.array(ordered)

# 1. 학습된 모델 로드
model = YOLO("runs/segment/runs/dacl10k/yolo11n_seg_1024-2/weights/best.pt")

# 2. RealSense 파이프라인 및 스트림 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

# ★ 추가: RealSense 하드웨어 필터 (Depth 데이터의 구멍을 주변값으로 메워줌)
hole_filling = rs.hole_filling_filter()

print("Intel RealSense 카메라 스트리밍 시작... (종료: 'q' 입력)")

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        # ★ 추가: Hole Filling 필터 적용
        aligned_depth_frame = hole_filling.process(aligned_depth_frame).as_depth_frame()
        depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        results = model(color_image, verbose=False)
        annotated_frame = results[0].plot()

        if results[0].masks is not None:
            for idx, polygon in enumerate(results[0].masks.xy):
                pts = np.array(polygon, np.int32)
                
                binary_mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(binary_mask, [pts], 255)

                thinned_mask = cv2.ximgproc.thinning(binary_mask, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
                y_coords, x_coords = np.where(thinned_mask > 0)
                skeleton_points = np.column_stack((x_coords, y_coords))
                
                if len(skeleton_points) > 1:
                    ordered_points = sort_skeleton_points(skeleton_points)
                    
                    # ★ 수정: 유효한 3D 좌표만 먼저 리스트로 추출 (단위: 미터)
                    valid_3d_points = []
                    for pt_2d in ordered_points:
                        x, y = int(pt_2d[0]), int(pt_2d[1])
                        
                        # raw 데이터(mm) 대신, 기기 스케일이 반영된 정확한 미터(m) 단위 측정
                        d_meters = aligned_depth_frame.get_distance(x, y)
                        
                        if d_meters > 0: # Depth가 0인 결측치는 건너뜀
                            pt3d_meters = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], d_meters)
                            valid_3d_points.append(pt3d_meters)
                    
                    # ★ 수정: 수집된 3D 점들을 연결하여 거리를 누적 (중간에 결측치가 있어도 건너뛰어 연결됨)
                    total_length_meters = 0.0
                    for i in range(1, len(valid_3d_points)):
                        p1 = valid_3d_points[i-1]
                        p2 = valid_3d_points[i]
                        
                        dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
                        total_length_meters += dist
                    
                    total_length_mm = total_length_meters * 1000  # 미터 -> 밀리미터 변환
                    
                    # 시각화 오버레이 (붉은 뼈대)
                    thinned_bgr = cv2.cvtColor(thinned_mask, cv2.COLOR_GRAY2BGR)
                    thinned_bgr[thinned_mask > 0] = [0, 0, 255]
                    cv2.addWeighted(annotated_frame, 1, thinned_bgr, 0.7, 0, annotated_frame)
                    
                    # 길이 텍스트 출력
                    x_min = np.min(pts[:, 0])
                    y_max = np.max(pts[:, 1])
                    if len(valid_3d_points) > 1:
                        cv2.putText(annotated_frame, f"Length: {total_length_mm:.1f}mm", (x_min, y_max + 25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        cv2.putText(annotated_frame, "Length: N/A", (x_min, y_max + 25), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 1. 팽창 연산을 위한 커널 생성 (숫자가 클수록 주변 벽면을 더 넓게 잡음)
            kernel = np.ones((15, 15), np.uint8)
            
            # 2. 균열 마스크 팽창 (균열 영역 + 주변 벽면 영역이 모두 포함됨)
            dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)
            
            # 3. 인접한 정상 벽면 마스크만 추출 (팽창된 마스크에서 원래 균열 마스크를 뺌)
            wall_mask = cv2.subtract(dilated_mask, binary_mask)
            
            # 4. 균열 내부와 주변 벽면의 Depth 데이터 분리 (밀리미터 단위)
            crack_depth_values = depth_image[binary_mask == 255]
            wall_depth_values = depth_image[wall_mask == 255]
            
            # 결측치(0)를 제외한 유효한 깊이 값만 필터링
            valid_crack_depths = crack_depth_values[crack_depth_values > 0]
            valid_wall_depths = wall_depth_values[wall_depth_values > 0]
            
            # 5. 두 영역의 깊이가 모두 유효하게 측정된 경우 단차 계산
            if len(valid_crack_depths) > 0 and len(valid_wall_depths) > 0:
                avg_crack_dist = np.mean(valid_crack_depths)  # 카메라부터 균열 안쪽까지 거리
                avg_wall_dist = np.mean(valid_wall_depths)    # 카메라부터 평평한 벽면까지 거리
                
                # 균열이 안으로 패어 있으므로, 카메라로부터의 거리는 균열이 더 멉니다.
                actual_crack_depth = avg_crack_dist - avg_wall_dist
                
                # 노이즈로 인해 마이너스 값이 나오는 것을 방지
                actual_crack_depth = max(0, actual_crack_depth)
            
                # 화면에 패인 깊이(Depth Diff) 출력 (기존 길이 텍스트 아래에 출력되도록 y좌표를 +45로 설정)
                cv2.putText(annotated_frame, f"Depth Diff: {actual_crack_depth:.1f}mm", 
                            (x_min, y_max + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((annotated_frame, depth_colormap))
        cv2.imshow('RealSense Crack Detection', images)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()