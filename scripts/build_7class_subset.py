from pathlib import Path
from PIL import Image
from collections import Counter
import json
import os
import random
import shutil


# --------------------------------------------------
# AYARLAR
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "deepfashion2_raw"

OUT_DIR = PROJECT_ROOT / "data" / "deepfashion2_yolo_7class"

TRAIN_LIMIT = 10000
VAL_LIMIT = 2000

SEED = 42


# --------------------------------------------------
# 13 DEEPFASHION2 SINIFI -> 7 PROJE SINIFI
# DeepFashion2 category_id değerleri 1-13
# Burada önce 0-12'ye çevrilip map ediliyor
# --------------------------------------------------

CLASS_MAP = {
    0: 0,   # short_sleeve_top -> top
    1: 0,   # long_sleeve_top -> top

    2: 1,   # short_sleeve_outwear -> outwear
    3: 1,   # long_sleeve_outwear -> outwear

    4: 2,   # vest -> sleeveless_top
    5: 2,   # sling -> sleeveless_top

    6: 3,   # shorts
    7: 4,   # trousers
    8: 5,   # skirt

    9: 6,   # short_sleeve_dress -> dress
    10: 6,  # long_sleeve_dress -> dress
    11: 6,  # vest_dress -> dress
    12: 6,  # sling_dress -> dress
}


CLASS_NAMES = [
    "top",
    "outwear",
    "sleeveless_top",
    "shorts",
    "trousers",
    "skirt",
    "dress",
]


# --------------------------------------------------
# POLYGON ALANI
# --------------------------------------------------

def polygon_area(points):

    if len(points) < 6:
        return 0.0

    coords = list(
        zip(
            points[0::2],
            points[1::2]
        )
    )

    area = 0.0

    for i in range(len(coords)):

        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % len(coords)]

        area += x1 * y2
        area -= x2 * y1

    return abs(area) / 2.0


# --------------------------------------------------
# SPLIT DÖNÜŞTÜR
# --------------------------------------------------

def convert_split(
    split_name,
    output_name,
    limit
):

    image_dir = RAW_DIR / split_name / "image"
    anno_dir = RAW_DIR / split_name / "annos"

    out_images = OUT_DIR / "images" / output_name
    out_labels = OUT_DIR / "labels" / output_name

    # Daha önce oluşmuşsa temizle
    if out_images.exists():
        shutil.rmtree(out_images)

    if out_labels.exists():
        shutil.rmtree(out_labels)

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    json_files = list(
        anno_dir.glob("*.json")
    )

    random.seed(SEED)
    random.shuffle(json_files)

    json_files = json_files[:limit]

    class_counts = Counter()

    converted_images = 0
    converted_objects = 0
    skipped_images = 0

    print(
        f"\n{split_name.upper()} hazırlanıyor..."
    )

    for index, json_path in enumerate(
        json_files,
        start=1
    ):

        image_path = (
            image_dir /
            f"{json_path.stem}.jpg"
        )

        if not image_path.exists():
            skipped_images += 1
            continue

        try:
            with Image.open(image_path) as img:
                width, height = img.size

        except Exception:
            skipped_images += 1
            continue

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        label_lines = []

        for key, item in data.items():

            if not key.startswith("item"):
                continue

            category_id = item.get(
                "category_id"
            )

            segments = item.get(
                "segmentation"
            )

            if category_id is None:
                continue

            if not segments:
                continue

            # DeepFashion2 1-13
            # -> 0-12
            old_class_id = (
                int(category_id) - 1
            )

            if old_class_id not in CLASS_MAP:
                continue

            # Yeni 7 sınıftan biri
            new_class_id = CLASS_MAP[
                old_class_id
            ]

            valid_segments = []

            for segment in segments:

                if not isinstance(
                    segment,
                    list
                ):
                    continue

                if len(segment) < 6:
                    continue

                # x,y çiftlerinin düzgün olması gerekir
                if len(segment) % 2 != 0:
                    continue

                valid_segments.append(
                    segment
                )

            if not valid_segments:
                continue

            # DeepFashion2 bazı nesnelerde
            # birden fazla polygon verebilir.
            # İlk aşamada ana/gövde polygonunu kullanıyoruz.
            segment = max(
                valid_segments,
                key=polygon_area
            )

            normalized = []

            for i in range(
                0,
                len(segment),
                2
            ):

                x = segment[i] / width
                y = segment[i + 1] / height

                # YOLO değerlerini güvenli şekilde
                # 0-1 aralığında tut
                x = max(
                    0.0,
                    min(1.0, x)
                )

                y = max(
                    0.0,
                    min(1.0, y)
                )

                normalized.extend(
                    [x, y]
                )

            if len(normalized) < 6:
                continue

            coords = " ".join(
                f"{value:.6f}"
                for value in normalized
            )

            label_lines.append(
                f"{new_class_id} {coords}"
            )

            class_counts[
                new_class_id
            ] += 1

            converted_objects += 1

        # Hiç kullanılabilir kıyafet yoksa
        # görüntüyü dataset'e ekleme
        if not label_lines:
            skipped_images += 1
            continue

        label_path = (
            out_labels /
            f"{json_path.stem}.txt"
        )

        with open(
            label_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "\n".join(label_lines)
            )

        target_image = (
            out_images /
            image_path.name
        )

        # Fotoğrafı ikinci kez kopyalamıyoruz.
        # Symbolic link oluşturuyoruz.
        if not target_image.exists():

            os.symlink(
                image_path.resolve(),
                target_image
            )

        converted_images += 1

        if index % 500 == 0:

            print(
                f"{index}/{len(json_files)} "
                f"işlendi"
            )

    print("\n" + "=" * 55)

    print(
        f"{output_name}: "
        f"{converted_images} görüntü"
    )

    print(
        f"{converted_objects} "
        f"kıyafet instance"
    )

    print(
        f"{skipped_images} görüntü atlandı"
    )

    print("\n7 sınıflı dağılım:\n")

    total = sum(
        class_counts.values()
    )

    for class_id, name in enumerate(
        CLASS_NAMES
    ):

        count = class_counts[
            class_id
        ]

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        print(
            f"{class_id} "
            f"{name:20s} "
            f"{count:6d} "
            f"{percentage:6.2f}%"
        )

    print("=" * 55)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    convert_split(
        split_name="train",
        output_name="train",
        limit=TRAIN_LIMIT
    )

    convert_split(
        split_name="validation",
        output_name="val",
        limit=VAL_LIMIT
    )

    print(
        "\n7 sınıflı dataset hazır."
    )