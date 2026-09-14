from ultralytics import YOLO

MODEL_PATH = "runs/segment/runs/deepfashion2_7class/test_3epoch-2/weights/best.pt"

model = YOLO(MODEL_PATH)

results = model.predict(
    source="test_images/shelf_test.jpg",
    device="mps",
    conf=0.05,
    iou=0.50,
    imgsz=1024,
    max_det=300,
    save=True
)

print("\nTAHMİNLER\n")

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            f"{model.names[class_id]} "
            f"{confidence:.2%}"
        )

print("\nSonuç:", results[0].save_dir)