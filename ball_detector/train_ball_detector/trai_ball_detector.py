from ultralytics import YOLO

model = YOLO("yolo11s.pt")

model.train(
    data="/content/Ball_dataset/data.yaml",
    imgsz=1280,
    epochs=20,
    batch=8,
    device = "cuda",
    project="/content/drive/MyDrive/yolo_runs",
    name="ball_detector",
    exist_ok=True
)

model = YOLO("/content/drive/MyDrive/yolo_runs/ball_detector/weights/best_weights_ball_detector.pt")
results = model.predict(
    source='/content/test_video.mp4',
    save=True,
    stream = True,
    project='/content/drive/MyDrive/YOLO_predictions',
    name='video_results',
    device = "cuda",
    conf=0.25
)

for r in results:
    pass