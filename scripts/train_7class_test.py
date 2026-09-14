from ultralytics import YOLO
import torch

print("MPS:", torch.backends.mps.is_available())

model = YOLO("yolo26l-seg.pt")

model.train(
    data="deepfashion2_7class.yaml",
    epochs=3,
    imgsz=512,
    batch=1,
    device="mps",
    workers=0,
    project="runs/deepfashion2_7class",
    name="test_3epoch",
    patience=3
)

print("Test eğitimi tamamlandı.")