"""Convert Fashionpedia COCO garment polygons to the project's seven YOLO classes."""
from collections import defaultdict
from pathlib import Path
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image
from ultralytics.data.converter import merge_multi_segment

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/fashionpedia"
OUTPUT = ROOT / "data/fashionpedia_yolo_7class"

# Fashionpedia main garment name -> project class ID.
# Jumpsuits and garment parts/accessories are intentionally excluded.
CLASS_MAP = {
    "shirt, blouse": 0,
    "top, t-shirt, sweatshirt": 0,
    "sweater": 0,
    "cardigan": 1,
    "jacket": 1,
    "coat": 1,
    "cape": 1,
    "vest": 2,
    "shorts": 3,
    "pants": 4,
    "skirt": 5,
    "dress": 6,
}


def polygon_area(points):
    pairs = list(zip(points[::2], points[1::2]))
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2)
                   in zip(pairs, pairs[1:] + pairs[:1]))) / 2


def decode_compressed_rle(counts):
    """Decode COCO's compressed run-length string without requiring pycocotools."""
    runs = []
    position = 0
    while position < len(counts):
        value = 0
        shift = 0
        while True:
            code = ord(counts[position]) - 48
            position += 1
            value |= (code & 0x1F) << (5 * shift)
            shift += 1
            if not code & 0x20:
                if code & 0x10:
                    value |= -1 << (5 * shift)
                break
        if len(runs) > 2:
            value += runs[-2]
        runs.append(value)
    return runs


def rle_to_polygons(segmentation):
    """Convert a compressed or uncompressed COCO RLE mask to outer polygons."""
    height, width = segmentation["size"]
    counts = segmentation["counts"]
    if isinstance(counts, str):
        counts = decode_compressed_rle(counts)
    flat = np.zeros(height * width, dtype=np.uint8)
    offset = 0
    foreground = False
    for run in counts:
        end = offset + run
        if foreground:
            flat[offset:end] = 1
        offset = end
        foreground = not foreground
    if offset != flat.size:
        raise ValueError(f"RLE size mismatch: decoded {offset}, expected {flat.size}")
    mask = flat.reshape((height, width), order="F")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour.reshape(-1, 2).astype(float).ravel().tolist()
            for contour in contours if len(contour) >= 3]


def image_index(source):
    index = defaultdict(list)
    for path in source.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            index[path.name].append(path)
    return index


def convert_split(split, annotation_path, source, output, overwrite=False):
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    category_names = {item["id"]: item["name"] for item in data["categories"]}
    target_category_ids = {category_id: CLASS_MAP[name] for category_id, name in category_names.items()
                           if name in CLASS_MAP}
    annotations = defaultdict(list)
    for item in data["annotations"]:
        if item["category_id"] in target_category_ids:
            annotations[item["image_id"]].append(item)

    indexed = image_index(source)
    image_out = output / "images" / split
    label_out = output / "labels" / split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    merged_segments = 0

    for image in data["images"]:
        items = annotations.get(image["id"], [])
        if not items:
            continue
        candidates = indexed.get(Path(image["file_name"]).name, [])
        if len(candidates) != 1:
            raise FileNotFoundError(f"Expected one image for {image['file_name']}, found {len(candidates)}")
        source_image = candidates[0]
        with Image.open(source_image) as opened:
            width, height = opened.size
        if (width, height) != (image["width"], image["height"]):
            raise ValueError(f"COCO/image size mismatch: {source_image}")

        lines = []
        for item in items:
            segments = item.get("segmentation")
            if isinstance(segments, dict):
                segments = rle_to_polygons(segments)
            elif not isinstance(segments, list):
                raise ValueError(f"Unsupported mask: annotation {item['id']}")
            valid = [segment for segment in segments
                     if isinstance(segment, list) and len(segment) >= 6 and len(segment) % 2 == 0]
            if not valid:
                raise ValueError(f"Missing polygon: annotation {item['id']}")
            if len(valid) > 1:
                merged_segments += len(valid) - 1
                segment = np.concatenate(merge_multi_segment(valid)).ravel().tolist()
            else:
                segment = valid[0]
            coords = [value / (width if index % 2 == 0 else height)
                      for index, value in enumerate(segment)]
            if not all(0 <= value <= 1 for value in coords):
                raise ValueError(f"Out-of-range polygon: annotation {item['id']}")
            class_id = target_category_ids[item["category_id"]]
            lines.append(str(class_id) + " " + " ".join(f"{value:.6f}" for value in coords))
            counts[class_id] += 1

        stem = f"fashionpedia_{source_image.stem}"
        target_image = image_out / f"{stem}{source_image.suffix.lower()}"
        target_label = label_out / f"{stem}.txt"
        label_text = "\n".join(lines)
        if (target_image.exists() or target_label.exists()) and not overwrite:
            link_ok = target_image.is_symlink() and target_image.resolve() == source_image.resolve()
            label_ok = target_label.is_file() and target_label.read_text(encoding="utf-8") == label_text
            if not (link_ok and label_ok):
                raise FileExistsError(f"Refusing to replace existing output: {target_image}")
        else:
            if not target_image.exists():
                os.symlink(source_image.resolve(), target_image)
            target_label.write_text(label_text, encoding="utf-8")

    return counts, merged_segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    splits = {
        "train": source / "instances_attributes_train2020.json",
        "val": source / "instances_attributes_val2020.json",
    }
    total_merged = 0
    for split, annotation in splits.items():
        if not annotation.is_file():
            raise FileNotFoundError(annotation)
        counts, merged = convert_split(split, annotation, source, output, args.overwrite)
        total_merged += merged
        print(split, dict(sorted(counts.items())))
    print(f"Merged secondary polygon parts: {total_merged}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
