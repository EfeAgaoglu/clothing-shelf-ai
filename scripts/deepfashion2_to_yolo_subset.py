from pathlib import Path
from PIL import Image
import json
import os

# --------------------------------------------------
# AYARLAR
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "deepfashion2_raw"
OUT_DIR = PROJECT_ROOT / "data" / "deepfashion2_yolo"

TRAIN_LIMIT = 500
VAL_LIMIT = 100


# --------------------------------------------------
# Polygon alanı
# Birden fazla segmentation polygon varsa
# en büyük olanı seçmek için kullanıyoruz.
# --------------------------------------------------

def polygon_area(points):
    if len(points) < 6:
        return 0

    coords = list(zip(points[0::2], points[1::2]))

    area = 0.0

    for i in range(len(coords)):
        x1, y1 = coords[i]
        x2, y2 = coords[(i + 1) % len(coords)]

        area += x1 * y2
        area -= x2 * y1

    return abs(area) / 2.0


# --------------------------------------------------
# Tek split dönüştürme
# --------------------------------------------------

def convert_split(split_name, output_name, limit):

    image_dir = RAW_DIR / split_name / "image"
    anno_dir = RAW_DIR / split_name / "annos"

    out_images = OUT_DIR / "images" / output_name
    out_labels = OUT_DIR / "labels" / output_name

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    json_files = sorted(anno_dir.glob("*.json"))[:limit]

    converted_images = 0
    converted_objects = 0

    print(f"\n{split_name} dönüştürülüyor...")

    for index, json_path in enumerate(json_files, start=1):

        image_path = image_dir / f"{json_path.stem}.jpg"

        if not image_path.exists():
            print("Görüntü bulunamadı:", image_path)
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        with open(json_path, "r") as f:
            data = json.load(f)

        label_lines = []

        for key, item in data.items():

            if not key.startswith("item"):
                continue

            category_id = item.get("category_id")
            segments = item.get("segmentation")

            if category_id is None or not segments:
                continue

            # DeepFashion2: 1-13
            # YOLO: 0-12
            class_id = int(category_id) - 1

            valid_segments = []

            for segment in segments:

                if not isinstance(segment, list):
                    continue

                if len(segment) < 6:
                    continue

                valid_segments.append(segment)

            if not valid_segments:
                continue

            # Şimdilik en büyük polygonu kullanıyoruz
            segment = max(valid_segments, key=polygon_area)

            normalized = []

            for i in range(0, len(segment), 2):

                x = segment[i] / width
                y = segment[i + 1] / height

                # Güvenlik için 0-1 aralığı
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))

                normalized.extend([x, y])

            coords = " ".join(f"{v:.6f}" for v in normalized)

            label_lines.append(
                f"{class_id} {coords}"
            )

            converted_objects += 1

        # Label dosyasını oluştur
        label_path = out_labels / f"{json_path.stem}.txt"

        with open(label_path, "w") as f:
            f.write("\n".join(label_lines))

        # Görüntüyü kopyalamak yerine symbolic link oluştur
        target_image = out_images / image_path.name

        if not target_image.exists():
            os.symlink(
                image_path.resolve(),
                target_image
            )

        converted_images += 1

        if index % 100 == 0:
            print(
                f"{index}/{len(json_files)} işlendi"
            )

    print(
        f"{output_name}: "
        f"{converted_images} görüntü, "
        f"{converted_objects} kıyafet"
    )


# --------------------------------------------------
# Çalıştır
# --------------------------------------------------

if __name__ == "__main__":

    convert_split(
        "train",
        "train",
        TRAIN_LIMIT
    )

    convert_split(
        "validation",
        "val",
        VAL_LIMIT
    )

    print("\nDönüştürme tamamlandı.")