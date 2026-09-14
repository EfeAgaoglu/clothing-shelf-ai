from pathlib import Path
import json
from collections import Counter
import csv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "deepfashion2_raw"

CLASS_NAMES = [
    "short_sleeve_top",
    "long_sleeve_top",
    "short_sleeve_outwear",
    "long_sleeve_outwear",
    "vest",
    "sling",
    "shorts",
    "trousers",
    "skirt",
    "short_sleeve_dress",
    "long_sleeve_dress",
    "vest_dress",
    "sling_dress",
]


def analyze_split(split_name):

    anno_dir = RAW_DIR / split_name / "annos"

    json_files = sorted(anno_dir.glob("*.json"))

    print(f"\n{'=' * 60}")
    print(f"{split_name.upper()} ANALİZİ")
    print(f"{'=' * 60}")

    print(f"Annotation dosyası: {len(json_files):,}")

    # Toplam nesne sayısı
    instance_counts = Counter()

    # Kaç farklı görüntüde göründüğü
    image_counts = Counter()

    total_objects = 0
    invalid_objects = 0

    for index, json_path in enumerate(json_files, start=1):

        try:
            with open(json_path, "r") as f:
                data = json.load(f)

        except Exception as e:
            print(f"Hatalı JSON: {json_path} -> {e}")
            continue

        classes_in_this_image = set()

        for key, item in data.items():

            if not key.startswith("item"):
                continue

            category_id = item.get("category_id")

            if category_id is None:
                invalid_objects += 1
                continue

            class_id = int(category_id) - 1

            if not 0 <= class_id < len(CLASS_NAMES):
                invalid_objects += 1
                continue

            instance_counts[class_id] += 1
            classes_in_this_image.add(class_id)

            total_objects += 1

        # Aynı sınıftan bir görüntüde 3 ürün olsa bile
        # image_counts yalnızca 1 artar
        for class_id in classes_in_this_image:
            image_counts[class_id] += 1

        if index % 25000 == 0:
            print(
                f"{index:,}/{len(json_files):,} işlendi..."
            )

    print("\nSonuç:")
    print(f"Toplam kıyafet: {total_objects:,}")
    print(f"Geçersiz nesne: {invalid_objects:,}")

    print("\nSınıf dağılımı:\n")

    print(
        f"{'ID':<4}"
        f"{'Sınıf':<27}"
        f"{'Instance':>12}"
        f"{'Görüntü':>12}"
        f"{'%':>9}"
    )

    print("-" * 64)

    results = []

    for class_id, class_name in enumerate(CLASS_NAMES):

        instances = instance_counts[class_id]
        images = image_counts[class_id]

        percentage = (
            instances / total_objects * 100
            if total_objects
            else 0
        )

        print(
            f"{class_id:<4}"
            f"{class_name:<27}"
            f"{instances:>12,}"
            f"{images:>12,}"
            f"{percentage:>8.2f}%"
        )

        results.append(
            {
                "split": split_name,
                "class_id": class_id,
                "class_name": class_name,
                "instances": instances,
                "images": images,
                "percentage": round(percentage, 4),
            }
        )

    return results


if __name__ == "__main__":

    all_results = []

    all_results.extend(
        analyze_split("train")
    )

    all_results.extend(
        analyze_split("validation")
    )

    output_file = (
        PROJECT_ROOT /
        "deepfashion2_class_distribution.csv"
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "class_id",
                "class_name",
                "instances",
                "images",
                "percentage",
            ]
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 60)
    print("ANALİZ TAMAMLANDI")
    print("=" * 60)

    print(
        "\nCSV kaydedildi:",
        output_file
    )