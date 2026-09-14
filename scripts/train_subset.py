from ultralytics import YOLO
import torch

print("MPS kullanılabilir:", torch.backends.mps.is_available())

model = YOLO("yolo26l-seg.pt")

results = model.train(
    data="deepfashion2.yaml",

    epochs=3,

    imgsz=512,

    batch=1,

    device="mps",

    workers=0,

    project="runs/deepfashion2",

    name="yolo26l_seg_test",

    patience=3
)

print("Eğitim tamamlandı.")