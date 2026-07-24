from ultralytics import YOLO
import torch

# --- CUDA 환경 정상 인식 확인 ---
if torch.cuda.is_available():
    print(f"✅ GPU 인식 성공: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ 경고: CUDA를 찾을 수 없습니다! PyTorch가 CPU 버전으로 설치되어 있을 수 있습니다.")
# ---------------------------------

# 1. 세그멘테이션 베이스 모델 로드
model = YOLO("yolo11n-seg.pt")

print("공식 crack-seg 데이터셋을 자동으로 다운로드하고 학습을 시작합니다...")

# 2. 학습 실행 
results = model.train(
    data="crack-seg.yaml",  
    epochs=50,              
    imgsz=640,              
    batch=8,                
    device=0,               
    name="crack_seg_official",
    verbose=False,
    seed=42,                # <--- 난수 생성 시드 고정 (원하는 숫자로 변경 가능)
    deterministic=True      # <--- cuDNN 연산의 무작위성을 배제하여 완벽한 재현성 보장
)

print("\n학습이 완료되었습니다!")
print("최적의 가중치 파일은 runs/segment/crack_seg_official/weights/best.pt 에 있습니다.")