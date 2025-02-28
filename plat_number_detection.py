import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import Annotator
import datetime
from PIL import Image
import pytesseract
import re
import easyocr

model = YOLO('anpr-demo-model.pt')
names = model.names

source = 'anpr-demo-video.mp4'
source1 = 'plat_number_video.mp4'
source2 = 'anpr-demo-video.mp4'
reader = easyocr.Reader(['en'])

cap = cv2.VideoCapture(source1)
lebar = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
tinggi = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

def show_frame(frame):
    plt.imshow(frame)
    plt.show()

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (lebar, tinggi))
    results = model.predict(frame, show=False, imgsz=320, verbose=False)
    result = results[0]
    boxes = result.boxes


    if boxes:
        for box in boxes:
            box_xyxy = box.xyxy.cpu().tolist()[0]
            x1 = int(box_xyxy[0])
            y1 = int(box_xyxy[1])
            x2 = int(box_xyxy[2])
            y2 = int(box_xyxy[3])

            plat_number = frame[y1 : y2, x1 : x2]
            plat_number = cv2.cvtColor(plat_number, cv2.COLOR_BGR2RGB)
            plat_number_text = reader.readtext(plat_number, detail=0)

            if len(plat_number_text) > 0:
                plat_number_text = plat_number_text[0]
            else:
                plat_number_text = 'plat number'
            
            annotator = Annotator(frame, line_width=2, example='names')
            annotator.box_label(box_xyxy, plat_number_text, (255, 0, 255), False)

    cv2.imshow('tes', frame)
    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

