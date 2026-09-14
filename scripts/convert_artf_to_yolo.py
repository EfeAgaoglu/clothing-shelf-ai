"""Convert the downloaded aRTF COCO polygons to the project's 7-class YOLO format."""
from collections import defaultdict
from pathlib import Path
import argparse
import json
import os

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/aRTF"
OUTPUT = ROOT / "data/internet_shelf_7class"

# aRTF category IDs: shorts=1, tshirt=2.
# Project IDs: top=0, shorts=3.
SOURCES = {
    "shorts": (1, 3),
    "tshirts": (2, 0),
}


def polygon_area(points):
    pairs = list(zip(points[::2], points[1::2]))
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2)
                   in zip(pairs, pairs[1:] + pairs[:1]))) / 2


def find_annotation(category, split):
    matches = list((SOURCE / category).glob(f"{category}-{split}_resized_512x256/*.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {category}/{split} annotation, found {len(matches)}")
    return matches[0]


def convert(overwrite=False):
    counts = defaultdict(int)
    for split in ("train", "val", "test"):
        image_out = OUTPUT / "images" / split
        label_out = OUTPUT / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        expected = set()
        for category, (source_class, target_class) in SOURCES.items():
            annotation_path = find_annotation(category, split)
            data = json.loads(annotation_path.read_text(encoding="utf-8"))
            annotations = defaultdict(list)
            for item in data["annotations"]:
                annotations[item["image_id"]].append(item)

            for image in data["images"]:
                source_image = annotation_path.parent / image["file_name"]
                if not source_image.is_file():
                    raise FileNotFoundError(source_image)
                with Image.open(source_image) as opened:
                    width, height = opened.size
                if (width, height) != (image["width"], image["height"]):
                    raise ValueError(f"COCO/image size mismatch: {source_image}")

                stem = f"artf_{category}_{Path(image['file_name']).stem}"
                target_image = image_out / f"{stem}{source_image.suffix.lower()}"
                target_label = label_out / f"{stem}.txt"
                expected.add(target_image.name)

                lines = []
                for item in annotations[image["id"]]:
                    if item["category_id"] != source_class:
                        raise ValueError(f"Unexpected category in {annotation_path}: {item['category_id']}")
                    segments = [segment for segment in item.get("segmentation", [])
                                if isinstance(segment, list) and len(segment) >= 6 and len(segment) % 2 == 0]
                    if not segments:
                        raise ValueError(f"Missing polygon for image id {image['id']} in {annotation_path}")
                    segment = max(segments, key=polygon_area)
                    coords = [value / (width if index % 2 == 0 else height)
                              for index, value in enumerate(segment)]
                    if not all(0 <= value <= 1 for value in coords):
                        raise ValueError(f"Out-of-range polygon for image id {image['id']}")
                    lines.append(str(target_class) + " " + " ".join(f"{value:.6f}" for value in coords))

                if not lines:
                    raise ValueError(f"No target annotation for {source_image}")
                if (target_image.exists() or target_label.exists()) and not overwrite:
                    existing_link = target_image.is_symlink() and target_image.resolve() == source_image.resolve()
                    existing_label = target_label.is_file() and target_label.read_text(encoding="utf-8") == "\n".join(lines)
                    if not (existing_link and existing_label):
                        raise FileExistsError(f"Refusing to replace existing output: {target_image}")
                else:
                    if not target_image.exists():
                        os.symlink(source_image.resolve(), target_image)
                    target_label.write_text("\n".join(lines), encoding="utf-8")
                counts[(split, category)] += 1

        unexpected = {path.name for path in image_out.iterdir()} - expected
        if unexpected:
            raise ValueError(f"Unexpected existing files in {image_out}: {sorted(unexpected)[:5]}")

    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    counts = convert(overwrite=args.overwrite)
    for split in ("train", "val", "test"):
        print(f"{split}: top={counts[(split, 'tshirts')]}, shorts={counts[(split, 'shorts')]}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
