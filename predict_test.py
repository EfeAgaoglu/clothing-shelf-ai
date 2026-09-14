from ultralytics import YOLO
import torch

print("MPS:", torch.backends.mps.is_available())

# YOLO26 Large Segmentation modelini yükle
model = YOLO("yolo26l-seg.pt")

# Fotoğraf üzerinde tahmin
results = model.predict(
    source="test_images/test.jpg",
    device="mps",
    conf=0.25,
    save=True
)

print("Tahmin tamamlandı.")
print("Sonuç klasörü:", results[0].save_dir)