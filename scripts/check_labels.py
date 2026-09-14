from pathlib import Path
from PIL import Image, ImageDraw
import random

ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = ROOT / "data/deepfashion2_yolo/images/train"
LABEL_DIR = ROOT / "data/deepfashion2_yolo/labels/train"
OUTPUT_DIR = ROOT / "debug_labels"

OUTPUT_DIR.mkdir(exist_ok=True)

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

images = list(IMAGE_DIR.glob("*.jpg"))

random.seed(42)
samples = random.sample(images, min(5, len(images)))

for image_path in samples:

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    width, height = image.size

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:

        values = line.strip().split()

        if len(values) < 7:
            continue

        class_id = int(values[0])
        coords = list(map(float, values[1:]))

        points = []

        for i in range(0, len(coords), 2):

            x = coords[i] * width
            y = coords[i + 1] * height

            points.append((x, y))

        if len(points) >= 3:

            draw.polygon(
                points,
                outline="red",
                width=3
            )

            x, y = points[0]

            draw.text(
                (x, y),
                CLASS_NAMES[class_id],
                fill="red"
            )

    output_path = OUTPUT_DIR / image_path.name

    image.save(output_path)

    print("Kaydedildi:", output_path)

print("\nKontrol görüntüleri hazır.")