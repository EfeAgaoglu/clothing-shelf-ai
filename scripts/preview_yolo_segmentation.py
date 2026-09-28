"""Render a small contact sheet of YOLO segmentation labels for visual QA."""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="shelf.yaml")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--output", default="runs/label_previews/preview.jpg")
    args = parser.parse_args()

    config_path = (ROOT / args.data).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_root = (config_path.parent / config["path"]).resolve()
    image_dir = dataset_root / config[args.split]
    label_dir = dataset_root / "labels" / image_dir.name
    names = config["names"]
    names = {int(key): value for key, value in names.items()} if isinstance(names, dict) else dict(enumerate(names))
    colors = {
        0: "#00e5ff", 1: "#ff7043", 2: "#ab47bc", 3: "#ffcc00",
        4: "#66bb6a", 5: "#ec407a", 6: "#42a5f5",
    }

    selected = []
    image_by_stem = {path.stem: path for path in sorted(image_dir.iterdir()) if path.is_file()}
    uncovered = set(names)
    for label_path in sorted(label_dir.glob("*.txt")):
        class_ids = {int(line.split()[0]) for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        if class_ids & uncovered and label_path.stem in image_by_stem:
            selected.append(image_by_stem[label_path.stem])
            uncovered -= class_ids
        if len(selected) >= args.count or not uncovered:
            break
    if len(selected) < args.count:
        selected_stems = {path.stem for path in selected}
        selected.extend(path for stem, path in image_by_stem.items() if stem not in selected_stems)
    selected = selected[:args.count]
    if not selected:
        parser.error(f"No images found in {image_dir}")

    tiles = []
    for image_path in selected:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        width, height = image.size
        for line in (label_dir / f"{image_path.stem}.txt").read_text(encoding="utf-8").splitlines():
            values = line.split()
            class_id = int(values[0])
            coords = list(map(float, values[1:]))
            points = [(coords[i] * width, coords[i + 1] * height) for i in range(0, len(coords), 2)]
            color = colors.get(class_id, "#ff00ff")
            draw.polygon(points, fill=color + "55", outline=color, width=2)
            draw.rectangle((4, 4, 120, 22), fill="black")
            draw.text((8, 6), names[class_id], fill="white")
        image.thumbnail((512, 256), Image.Resampling.LANCZOS)
        tiles.append(image)

    columns = 2
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (512 * columns, 256 * rows), "white")
    for index, tile in enumerate(tiles):
        x = (index % columns) * 512 + (512 - tile.width) // 2
        y = (index // columns) * 256 + (256 - tile.height) // 2
        sheet.paste(tile, (x, y))
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    print(output)


if __name__ == "__main__":
    main()
