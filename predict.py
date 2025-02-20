from ultralytics import YOLO

model = YOLO('yolo11n.pt')

result = model.predict('sample-illegal-parking.mp4', show=True, imgsz=640)