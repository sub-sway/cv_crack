import os
import json
import math
import shutil
import random
from collections import Counter, defaultdict
from tqdm import tqdm


def is_valid_polygon(seg):
    """YOLO segmentation polygon으로 사용 가능한지 검사"""
    if not isinstance(seg, list):
        return False

    if len(seg) < 6:
        return False

    if len(seg) % 2 != 0:
        return False

    if not all(math.isfinite(value) for value in seg):
        return False

    return True


def process_dacl10k_to_yolo(
    coco_json_path,
    image_source_dir,
    output_root,
    target_classes,
    val_ratio=0.2,
    seed=42
):
    # -------------------------------------------------
    # 1. 입력 경로 확인
    # -------------------------------------------------
    if not os.path.isfile(coco_json_path):
        raise FileNotFoundError(
            f"COCO JSON 파일을 찾을 수 없습니다.\n"
            f"입력 경로: {coco_json_path}\n"
            f"절대 경로: {os.path.abspath(coco_json_path)}"
        )

    if not os.path.isdir(image_source_dir):
        raise FileNotFoundError(
            f"이미지 폴더를 찾을 수 없습니다.\n"
            f"입력 경로: {image_source_dir}\n"
            f"절대 경로: {os.path.abspath(image_source_dir)}"
        )

    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio는 0과 1 사이여야 합니다.")

    # -------------------------------------------------
    # 2. 출력 폴더 생성
    # -------------------------------------------------
    for split in ["train", "val"]:
        os.makedirs(
            os.path.join(output_root, "images", split),
            exist_ok=True
        )
        os.makedirs(
            os.path.join(output_root, "labels", split),
            exist_ok=True
        )

    # -------------------------------------------------
    # 3. COCO JSON 로드
    # -------------------------------------------------
    print(f"\n데이터 로드: {coco_json_path}")

    with open(coco_json_path, "r", encoding="utf-8") as file:
        coco_data = json.load(file)

    required_keys = ["images", "annotations", "categories"]

    for key in required_keys:
        if key not in coco_data:
            raise KeyError(
                f"COCO JSON에 '{key}' 항목이 없습니다."
            )

    # -------------------------------------------------
    # 4. 클래스 매핑
    # -------------------------------------------------
    name_to_coco_id = {
        category["name"].strip().lower(): category["id"]
        for category in coco_data["categories"]
    }

    print("\nCOCO 데이터셋 클래스 목록")
    for category in coco_data["categories"]:
        print(
            f"- {category['name']} "
            f"(COCO ID: {category['id']})"
        )

    missing_classes = [
        class_name
        for class_name in target_classes
        if class_name.strip().lower() not in name_to_coco_id
    ]

    if missing_classes:
        raise ValueError(
            f"\n다음 클래스를 COCO categories에서 찾지 못했습니다: "
            f"{missing_classes}\n"
            f"위에 출력된 실제 클래스 이름과 철자를 맞춰주세요."
        )

    target_mapping = {
        name_to_coco_id[class_name.strip().lower()]: yolo_id
        for yolo_id, class_name in enumerate(target_classes)
    }

    print("\n타겟 클래스 매핑")

    for coco_id, yolo_id in target_mapping.items():
        print(
            f"- {target_classes[yolo_id]}: "
            f"COCO ID {coco_id} -> YOLO ID {yolo_id}"
        )

    # -------------------------------------------------
    # 5. 이미지 파일 전체 스캔
    # -------------------------------------------------
    print("\n원본 이미지 파일 스캔 중...")

    image_paths = {}
    duplicate_image_names = defaultdict(list)

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    )

    for root, _, files in os.walk(image_source_dir):
        for filename in files:
            if not filename.lower().endswith(valid_extensions):
                continue

            full_path = os.path.join(root, filename)

            if filename in image_paths:
                duplicate_image_names[filename].append(
                    image_paths[filename]
                )
                duplicate_image_names[filename].append(full_path)
            else:
                image_paths[filename] = full_path

    if duplicate_image_names:
        duplicate_examples = list(
            duplicate_image_names.items()
        )[:10]

        raise ValueError(
            "서로 다른 폴더에 같은 이미지 파일명이 존재합니다.\n"
            f"중복 예시: {duplicate_examples}\n"
            "현재 방식은 파일명만으로 이미지를 찾기 때문에 "
            "잘못된 이미지와 연결될 수 있습니다."
        )

    print(f"스캔된 이미지 수: {len(image_paths)}")

    # -------------------------------------------------
    # 6. COCO 이미지 정보 구성
    # -------------------------------------------------
    images_dict = {}

    for image in coco_data["images"]:
        image_id = image["id"]

        # COCO file_name에 하위 경로가 포함된 경우를 대비
        filename = os.path.basename(image["file_name"])

        images_dict[image_id] = {
            "filename": filename,
            "width": image["width"],
            "height": image["height"]
        }

    yolo_labels = {
        image_id: []
        for image_id in images_dict
    }

    # -------------------------------------------------
    # 7. Annotation 변환
    # -------------------------------------------------
    class_annotation_count = Counter()
    class_image_ids = defaultdict(set)

    invalid_polygon_count = 0
    rle_count = 0
    missing_image_id_count = 0

    print("\nPolygon 좌표 변환 중...")

    for annotation in tqdm(
        coco_data["annotations"],
        desc="Annotation 변환"
    ):
        category_id = annotation.get("category_id")

        if category_id not in target_mapping:
            continue

        image_id = annotation.get("image_id")

        if image_id not in images_dict:
            missing_image_id_count += 1
            continue

        image_info = images_dict[image_id]
        image_width = image_info["width"]
        image_height = image_info["height"]

        if image_width <= 0 or image_height <= 0:
            continue

        yolo_id = target_mapping[category_id]
        segmentation = annotation.get("segmentation")

        # RLE 형식은 별도 변환 필요
        if isinstance(segmentation, dict):
            rle_count += 1
            continue

        if not isinstance(segmentation, list):
            continue

        for polygon in segmentation:
            if not is_valid_polygon(polygon):
                invalid_polygon_count += 1
                continue

            normalized_coords = []

            for index in range(0, len(polygon), 2):
                x = polygon[index]
                y = polygon[index + 1]

                x_norm = max(
                    0.0,
                    min(1.0, x / image_width)
                )

                y_norm = max(
                    0.0,
                    min(1.0, y / image_height)
                )

                normalized_coords.extend(
                    [x_norm, y_norm]
                )

            coords_string = " ".join(
                f"{coordinate:.6f}"
                for coordinate in normalized_coords
            )

            yolo_labels[image_id].append(
                f"{yolo_id} {coords_string}"
            )

            class_annotation_count[yolo_id] += 1
            class_image_ids[yolo_id].add(image_id)

    # -------------------------------------------------
    # 8. 실제 존재하는 이미지만 분할 대상에 포함
    # -------------------------------------------------
    available_image_ids = []

    missing_files = []

    for image_id, image_info in images_dict.items():
        filename = image_info["filename"]

        if filename in image_paths:
            available_image_ids.append(image_id)
        else:
            missing_files.append(filename)

    print(f"\nJSON 이미지 수: {len(images_dict)}")
    print(f"실제 발견된 대응 이미지 수: {len(available_image_ids)}")
    print(f"찾지 못한 이미지 수: {len(missing_files)}")

    if missing_files:
        print(
            f"찾지 못한 이미지 예시: {missing_files[:20]}"
        )

    if not available_image_ids:
        raise RuntimeError(
            "JSON의 이미지 파일명과 실제 이미지가 하나도 연결되지 않았습니다."
        )

    # -------------------------------------------------
    # 9. Train / Validation 분할
    # -------------------------------------------------
    random_generator = random.Random(seed)
    random_generator.shuffle(available_image_ids)

    val_size = int(
        len(available_image_ids) * val_ratio
    )

    val_ids = set(available_image_ids[:val_size])
    train_ids = set(available_image_ids[val_size:])

    print(
        f"\nTrain: {len(train_ids)}장, "
        f"Validation: {len(val_ids)}장"
    )

    # -------------------------------------------------
    # 10. 이미지 복사 및 라벨 저장
    # -------------------------------------------------
    copied_count = 0

    for image_id in tqdm(
        available_image_ids,
        desc="YOLO 데이터셋 생성"
    ):
        split = "val" if image_id in val_ids else "train"

        image_info = images_dict[image_id]
        filename = image_info["filename"]

        source_image_path = image_paths[filename]

        destination_image_path = os.path.join(
            output_root,
            "images",
            split,
            filename
        )

        label_filename = (
            os.path.splitext(filename)[0] + ".txt"
        )

        destination_label_path = os.path.join(
            output_root,
            "labels",
            split,
            label_filename
        )

        shutil.copy2(
            source_image_path,
            destination_image_path
        )

        with open(
            destination_label_path,
            "w",
            encoding="utf-8"
        ) as file:
            label_lines = yolo_labels[image_id]

            if label_lines:
                file.write(
                    "\n".join(label_lines) + "\n"
                )

        copied_count += 1

    # -------------------------------------------------
    # 11. 결과 통계
    # -------------------------------------------------
    positive_images = sum(
        1
        for image_id in available_image_ids
        if yolo_labels[image_id]
    )

    empty_images = (
        len(available_image_ids) - positive_images
    )

    print("\n변환 완료")
    print(f"- 복사된 이미지: {copied_count}")
    print(f"- 결함 포함 이미지: {positive_images}")
    print(f"- 빈 라벨 이미지: {empty_images}")
    print(f"- 잘못된 polygon: {invalid_polygon_count}")
    print(f"- RLE annotation: {rle_count}")
    print(
        f"- 존재하지 않는 image_id annotation: "
        f"{missing_image_id_count}"
    )

    print("\n클래스별 통계")

    for yolo_id, class_name in enumerate(target_classes):
        print(
            f"- {class_name}: "
            f"polygon={class_annotation_count[yolo_id]}, "
            f"images={len(class_image_ids[yolo_id])}"
        )

    # -------------------------------------------------
    # 12. data.yaml 자동 생성
    # -------------------------------------------------
    yaml_path = os.path.join(
        output_root,
        "data.yaml"
    )

    absolute_output_root = os.path.abspath(output_root)

    with open(yaml_path, "w", encoding="utf-8") as file:
        file.write(
            f"path: {absolute_output_root}\n"
        )
        file.write("train: images/train\n")
        file.write("val: images/val\n\n")
        file.write("names:\n")

        for yolo_id, class_name in enumerate(target_classes):
            file.write(
                f"  {yolo_id}: {class_name}\n"
            )

    print(f"\ndata.yaml 생성 완료: {yaml_path}")
    print(f"최종 데이터셋 경로: {absolute_output_root}")


if __name__ == "__main__":
    # labels.json 실행 시 출력되는 실제 category 이름과
    # 아래 철자가 정확히 같아야 합니다.
    MY_TARGETS = [
        "Crack",
        "Spalling",
        "Efflorescence",
        "ExposedRebars"
    ]

    process_dacl10k_to_yolo(
        coco_json_path="./dacl10k_coco_labels/labels.json",
        image_source_dir="./dacl10k_raw/data",
        output_root="./dacl10k_yolo",
        target_classes=MY_TARGETS,
        val_ratio=0.2,
        seed=42
    )
