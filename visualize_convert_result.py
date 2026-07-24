import os
import cv2
import numpy as np
import random


def visualize_yolo_seg(
    image_dir,
    label_dir,
    output_dir,
    class_names,
    sample_count=50
):
    os.makedirs(output_dir, exist_ok=True)

    image_files = [
        file_name
        for file_name in os.listdir(image_dir)
        if file_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    selected_files = random.sample(
        image_files,
        min(sample_count, len(image_files))
    )

    for image_file in selected_files:
        image_path = os.path.join(
            image_dir,
            image_file
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        height, width = image.shape[:2]

        label_file = (
            os.path.splitext(image_file)[0] + ".txt"
        )

        label_path = os.path.join(
            label_dir,
            label_file
        )

        if os.path.exists(label_path):
            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as file:
                lines = file.readlines()

            for line in lines:
                values = line.strip().split()

                if len(values) < 7:
                    continue

                class_id = int(values[0])
                coordinates = list(
                    map(float, values[1:])
                )

                points = []

                for index in range(
                    0,
                    len(coordinates),
                    2
                ):
                    x = int(coordinates[index] * width)
                    y = int(
                        coordinates[index + 1] * height
                    )

                    points.append([x, y])

                points = np.array(
                    points,
                    dtype=np.int32
                )

                cv2.polylines(
                    image,
                    [points],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2
                )

                if len(points) > 0:
                    cv2.putText(
                        image,
                        class_names[class_id],
                        tuple(points[0]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        output_path = os.path.join(
            output_dir,
            image_file
        )

        cv2.imwrite(output_path, image)


visualize_yolo_seg(
    image_dir="dacl10k_yolo/images/train",
    label_dir="dacl10k_yolo/labels/train",
    output_dir="dacl10k_yolo/label_check",
    class_names=[
        "Crack",
        "Spalling",
        "Efflorescence",
        "ExposedRebar"
    ],
    sample_count=100
)