import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import Annotator
import datetime

def checkMousePosition(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        print(f'x:{x} | y:{y}')

def show_frame(frame):
    plt.imshow(frame)
    plt.show()

def show_obj_line(frame, line):
    line = np.array([line], np.int32)
    cv2.polylines(frame, [line], True, (0, 255, 0), 5)
    cv2.namedWindow('Parking Area')

model = YOLO('yolov8s.pt')
names = model.names
illegal_area = np.array([[538, 129], [742, 142], [794, 713], [131, 658]], np.int32)

source = 'sample-illegal-parking.mp4'

cap = cv2.VideoCapture(source)

time_in_boudary = {
    'time_detected': None,
    'max_time_in_boundary': None
}
while True:
    ret, frame = cap.read()
    results = model.predict(frame, show=False, imgsz=320)
    
    for result in results:
        classes = result.names
        boxes = result.boxes

        for box in boxes:
            class_id = box.cls[0].item()
            print(f'Class: {classes[class_id]}')
            break

    break

    cv2.namedWindow('Parking Area')
    cv2.setMouseCallback('Parking Area', checkMousePosition)

    cv2.polylines(frame, [illegal_area], True, (0, 0, 255), 2)
    cv2.namedWindow('Parking Area')
    cv2.imshow('Parking Area', frame)
    cv2.waitKey(0)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

# # Load a model
# model = YOLO('yolov11n.pt')  # load an official model

# # Predict on an image
# results = model('your_image.jpg')

# # Access bounding box data
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     for box in boxes:
#         # Get bounding box coordinates in (x, y, w, h) format
#         xywh = box.xywh[0].tolist()
#         class_id = int(box.cls[0].item())
#         confidence = box.conf[0].item()

#         print(f"Class: {class_id}, Coordinates: {xywh}, Confidence: {confidence}")