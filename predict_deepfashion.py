from ultralytics import YOLO
from pathlib import Path
import torch

print("MPS:", torch.backends.mps.is_available())

# Eğittiğimiz modeli bul
weights = list(Path("runs").rglob("best.pt"))

if not weights:
    raise FileNotFoundError("best.pt bulunamadı.")

model_path = weights[-1]

print("Kullanılan model:")
print(model_path)

model = YOLO(str(model_path))

results = model.predict(
    source="test_images/test.jpg",
    device="mps",
    conf=0.35,
    iou=0.50,
    agnostic_nms=True,
    imgsz=512,
    save=True
)

print("\nTahmin tamamlandı.")

for result in results:

    if result.boxes is None:
        continue

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            model.names[class_id],
            f"{confidence:.2%}"
        )

print("Sonuç:", results[0].save_dir)