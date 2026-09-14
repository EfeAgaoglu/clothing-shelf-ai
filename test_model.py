from ultralytics import YOLO
import torch

print("MPS aktif mi?:", torch.backends.mps.is_available())

model = YOLO("yolo26l-seg.pt")

print("YOLO26l-seg başarıyla yüklendi.")