"""Create a deterministic, class-balanced Fashionpedia subset using symlinks."""
from collections import Counter, deque
from pathlib import Path
import argparse
import json
import os
import random


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/fashionpedia_yolo_7class"
OUTPUT = ROOT / "data/fashionpedia_balanced_10k"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def class_ids(label_path):
    return {int(line.split()[0]) for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()}


def safe_link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() and target.resolve() == source.resolve():
        return
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to replace existing output: {target}")
    os.symlink(source.resolve(), target)


def link_split(source, output, split, stems):
    images = {path.stem: path for path in (source / "images" / split).iterdir()
              if path.suffix.lower() in IMAGE_SUFFIXES}
    for stem in stems:
        image = images[stem]
        label = source / "labels" / split / f"{stem}.txt"
        safe_link(image, output / "images" / split / image.name)
        safe_link(label, output / "labels" / split / label.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()

    labels = sorted((source / "labels" / "train").glob("*.txt"))
    labels_by_stem = {path.stem: path for path in labels}
    classes_by_stem = {stem: class_ids(path) for stem, path in labels_by_stem.items()}
    if not 0 < args.size <= len(classes_by_stem):
        parser.error(f"--size must be between 1 and {len(classes_by_stem)}")
    frequency = Counter(class_id for ids in classes_by_stem.values() for class_id in ids)

    # Assign each multi-label image to its rarest class, then sample the seven
    # groups round-robin. This prevents common top/dress labels from dominating.
    groups = {class_id: [] for class_id in range(7)}
    for stem, ids in classes_by_stem.items():
        if not ids:
            continue
        anchor = min(ids, key=lambda class_id: (frequency[class_id], class_id))
        groups[anchor].append(stem)
    rng = random.Random(args.seed)
    queues = {}
    for class_id, stems in groups.items():
        rng.shuffle(stems)
        queues[class_id] = deque(stems)

    selected, anchor_counts = [], Counter()
    while len(selected) < args.size:
        progressed = False
        for class_id in range(7):
            if queues[class_id] and len(selected) < args.size:
                selected.append(queues[class_id].popleft())
                anchor_counts[class_id] += 1
                progressed = True
        if not progressed:
            raise RuntimeError("Not enough labeled images to build the requested subset")

    link_split(source, output, "train", selected)
    val_stems = sorted(path.stem for path in (source / "labels" / "val").glob("*.txt"))
    link_split(source, output, "val", val_stems)

    instance_counts = Counter()
    image_counts = Counter()
    for stem in selected:
        lines = labels_by_stem[stem].read_text(encoding="utf-8").splitlines()
        ids = [int(line.split()[0]) for line in lines if line.strip()]
        instance_counts.update(ids)
        image_counts.update(set(ids))
    report = {
        "seed": args.seed,
        "train_images": len(selected),
        "val_images": len(val_stems),
        "anchor_image_counts": dict(sorted(anchor_counts.items())),
        "class_image_counts": dict(sorted(image_counts.items())),
        "class_instance_counts": dict(sorted(instance_counts.items())),
    }
    (output / "subset_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
