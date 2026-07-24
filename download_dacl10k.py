from huggingface_hub import snapshot_download
import os
import time

download_path = "./dacl10k_raw"
os.makedirs(download_path, exist_ok=True)

print("⏳ Hugging Face 데이터셋 다운로드를 시작합니다.")
print("무료 계정 제한(5분당 1000개)에 도달하면 자동으로 5분 대기 후 이어받습니다.")
print("창을 켜두고 다른 작업을 하셔도 좋습니다!\n")

while True:
    try:
        snapshot_download(
            repo_id="Voxel51/dacl10k",
            repo_type="dataset",
            local_dir=download_path,
            local_dir_use_symlinks=False
        )
        print("\n✨ 모든 데이터(약 9000장) 다운로드가 완벽하게 끝났습니다!")
        break  # 에러 없이 완료되면 루프 종료
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "Too Many Requests" in error_msg:
            print("\n🛑 [Hugging Face 5분 제한 도달]")
            print("이미 다운로드된 파일은 안전합니다. 5분(305초) 동안 휴식한 뒤 남은 파일을 자동으로 이어받습니다...")
            
            # 진행 상황을 보여주는 카운트다운
            for remaining in range(305, 0, -1):
                print(f"\r재시작까지 남은 시간: {remaining}초...", end="", flush=True)
                time.sleep(1)
            print("\n\n🚀 다운로드를 다시 시작합니다!")
        else:
            # 429 에러가 아닌 진짜 에러가 발생하면 중단
            print(f"\n❌ 알 수 없는 에러 발생: {error_msg}")
            break