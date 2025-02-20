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
    boxes = results[0].boxes.xyxy.cpu().tolist()
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
                if time_in_boudary.get('time_detected') is None:
                    now = datetime.datetime.now()
                    future_time = now + datetime.timedelta(seconds=10)
                    time_in_boudary.update({
                        'time_detected': now, 
                        'max_time_in_boundary': future_time
                    })

                vechiles.append('vechiles')

            cv2.namedWindow('Parking Area')

            annotator = Annotator(frame, line_width=2, example=names)
            identity = 'object'
            annotator.box_label(box, identity, (255, 0, 255), False)

    if not ret:
        break

    # illegal parking check
    if len(vechiles) > 0 and datetime.datetime.now() >= time_in_boudary.get('max_time_in_boundary'):
        print('max time in:', time_in_boudary.get('max_time_in_boundary'))
        print('current time:', datetime.datetime.now())
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

