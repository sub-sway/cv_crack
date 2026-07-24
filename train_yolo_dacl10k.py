from ultralytics import YOLO
import torch
import pandas as pd
import os


def main():
    # GPU 확인
    if torch.cuda.is_available():
        print(
            f"GPU 인식 성공: "
            f"{torch.cuda.get_device_name(0)}"
        )
        device = 0
    else:
        print("경고: CUDA를 찾을 수 없습니다.")
        device = "cpu"

    # 저장 설정
    project_dir = "runs/dacl10k"
    run_name = "yolo11n_seg_1024"

    # YOLO11s segmentation 모델
    model = YOLO("yolo11n-seg.pt")

    print("DACL10K 학습을 시작합니다.")

    results = model.train(
        data="dacl10k_yolo/data.yaml",

        epochs=100,
        imgsz=1024,
        batch=8,
        workers=8,
        device=device,

        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,

        patience=30,

        hsv_h=0.01,
        hsv_s=0.30,
        hsv_v=0.25,

        degrees=5.0,
        translate=0.05,
        scale=0.25,
        shear=2.0,
        perspective=0.0002,

        fliplr=0.5,
        flipud=0.0,

        mosaic=0.3,
        close_mosaic=15,

        mixup=0.0,
        copy_paste=0.0,

        cache=True,
        plots=True,
        save=True,

        project=project_dir,
        name=run_name
    )

    print("\n학습이 완료되었습니다.")

    # 실제 학습 결과 저장 경로 사용
    save_dir = str(results.save_dir)
    csv_path = os.path.join(save_dir, "results.csv")

    best_model_path = os.path.join(
        save_dir,
        "weights",
        "best.pt"
    )

    last_model_path = os.path.join(
        save_dir,
        "weights",
        "last.pt"
    )

    print(f"결과 저장 폴더: {save_dir}")
    print(f"최고 성능 모델: {best_model_path}")
    print(f"마지막 모델: {last_model_path}")

    # CSV 분석
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        print("\nCSV 컬럼 목록")
        print(df.columns.tolist())

        desired_columns = [
            "epoch",
            "train/seg_loss",
            "val/seg_loss",
            "metrics/mAP50(M)",
            "metrics/mAP50-95(M)"
        ]

        available_columns = [
            column
            for column in desired_columns
            if column in df.columns
        ]

        if available_columns:
            summary_df = df[available_columns].copy()

            pd.set_option(
                "display.float_format",
                "{:.4f}".format
            )

            print("\n학습 결과 요약")

            if len(summary_df) > 15:
                print(
                    summary_df.head(5).to_string(
                        index=False
                    )
                )
                print("... 중략 ...")
                print(
                    summary_df.tail(10).to_string(
                        index=False
                    )
                )
            else:
                print(
                    summary_df.to_string(index=False)
                )

            # Mask mAP@50 최고 Epoch
            if "metrics/mAP50(M)" in df.columns:
                best_index = df[
                    "metrics/mAP50(M)"
                ].idxmax()

                best_row = df.loc[best_index]

                print("\nMask mAP@50 기준 최고 결과")
                print(
                    f"Epoch: "
                    f"{int(best_row['epoch'])}"
                )
                print(
                    f"Mask mAP@50: "
                    f"{best_row['metrics/mAP50(M)']:.4f}"
                )

                if (
                    "metrics/mAP50-95(M)"
                    in df.columns
                ):
                    print(
                        f"Mask mAP@50-95: "
                        f"{best_row['metrics/mAP50-95(M)']:.4f}"
                    )
        else:
            print(
                "출력할 성능 컬럼을 찾지 못했습니다."
            )
    else:
        print(f"results.csv를 찾을 수 없습니다: {csv_path}")


if __name__ == "__main__":
    main()