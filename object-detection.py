import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import Annotator

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

model = YOLO('yolo11n.pt')
names = model.names
illegal_area = np.array([[78, 0], [258, 0], [273, 275], [46, 222]], np.int32)

source = 'illegal-parking.mp4'

cap = cv2.VideoCapture(source)

while True:
    ret, frame = cap.read()
    results = model.predict(frame, show=False, imgsz=320)
    boxes = results[0].boxes.xyxy.cpu().tolist()
    a=results[0].boxes.data
    vechiles = []

    if boxes:
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            line = np.array([[x1, y2], [x1, x2]], np.int32)
            cordinat_x=int(x1+x2)//2
            cordinat_y=int(y1+y2)//2
            center_object = (cordinat_x, cordinat_y)

            show_obj_line(frame, [cordinat_x, cordinat_y])

            test = cv2.pointPolygonTest(illegal_area, center_object,False)
            test = int(test)
            if test >= 0:
                vechiles.append('vechiles')

            cv2.namedWindow('Parking Area')

            annotator = Annotator(frame, line_width=2, example=names)
            identity = 'car'
            annotator.box_label(box, identity, (0, 0, 255), False)

    if not ret:
        break

    print('illegal parking', len(vechiles))

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

