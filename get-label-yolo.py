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

def is_vechile(class_name):
    result = False
    for vechile in vechiles:
        if class_name == vechile:
            result = True
            break

    return result

def setTextInFrame(frame, text, color):
    org = (50, 100) 
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = tuple(color)
    thickness = 2

    # Add text to the image
    cv2.putText(frame, text, org, font, fontScale, color, thickness)

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
    text_information = 'No illegal parking detected'
    color = [0, 255, 0]

    results = model.predict(frame, show=False, imgsz=320)
    result = results[0]
    vechiles_in_boundary = []
    vechiles = ['car', 'bicycle', 'bus', 'truck', 'motorcycle']

    classes = result.names
    boxes = result.boxes

    for box in boxes:
        class_id = box.cls[0].item()
        class_name = classes[class_id]

        if not is_vechile(class_name):
            continue

        box_xyxy = box.xyxy.cpu().tolist()[0]

        x1 = int(box_xyxy[0])
        y1 = int(box_xyxy[1])
        x2 = int(box_xyxy[2])
        y2 = int(box_xyxy[3])
        
        line = np.array([[x1, y2], [x1, x2]], np.int32)
        cordinat_x=int(x1+x2)//2
        cordinat_y=int(y1+y2)//2
        center_object = (cordinat_x, cordinat_y)

        car_in_boundary = -1
        show_obj_line(frame, [cordinat_x, cordinat_y])
        car_in_boundary = cv2.pointPolygonTest(illegal_area, center_object,False)
        car_in_boundary = int(car_in_boundary)

        if car_in_boundary >= 0:
            if time_in_boudary.get('time_detected') is None:
                now = datetime.datetime.now()
                future_time = now + datetime.timedelta(seconds=10)
                time_in_boudary.update({
                    'time_detected': now, 
                    'max_time_in_boundary': future_time
                })

            vechiles_in_boundary.append('vechiles')

        cv2.namedWindow('Parking Area')
                
        annotator = Annotator(frame, line_width=2, example=class_name)
        annotator.box_label(box_xyxy, class_name, (255, 0, 255), False)

    if len(vechiles) > 0 and time_in_boudary.get('max_time_in_boundary'):
        if len(vechiles) > 0 and datetime.datetime.now() >= time_in_boudary.get('max_time_in_boundary'):
            text_information = f'{len(vechiles_in_boundary)} vechiles Illegal parking detected'
            color = [0, 0, 255]
    else:
        color = [0, 255, 0]

    cv2.polylines(frame, [illegal_area], True, tuple(color), 2)
    setTextInFrame(frame, text_information, color)
    cv2.namedWindow('Parking Area')
    cv2.setMouseCallback('Parking Area', checkMousePosition)
    
    cv2.namedWindow('Parking Area')
    cv2.imshow('Parking Area', frame)
    # cv2.waitKey(0)
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