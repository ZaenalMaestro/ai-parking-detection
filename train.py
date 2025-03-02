from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    path = 'D:/MY-PROJECT/ilegal-parking/yolov8parkingspace/plat_number/data.yaml'
    results = model.train(data=path, epochs=100, imgsz=640, device=0)