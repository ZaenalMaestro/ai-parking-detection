from ultralytics import YOLO

model = YOLO('yolo11n.pt')

result = model.predict('parking1.mp4', show=True, imgsz=640)