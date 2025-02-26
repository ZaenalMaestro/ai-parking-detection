import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import Annotator
import datetime
from PIL import Image
import pytesseract
import re

model = YOLO('anpr-demo-model.pt')
names = model.names

source = 'anpr-demo-video.mp4'

cap = cv2.VideoCapture(source)
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
            # gray = cv2.cvtColor(plat_number, cv2.COLOR_BGR2RGB)
            # text_plat_number = pytesseract.image_to_string(gray)
            # text_plat_number = re.sub(r'[^\w\s]', '', text_plat_number) 
            # string_tanpa_spasi = "".join(text_plat_number.split())

            # if len(string_tanpa_spasi) == 0:
            #     continue
            annotator = Annotator(frame, line_width=2, example='names')
            annotator.box_label(box_xyxy, 'plat number', (255, 0, 255), False)

    cv2.imshow('Parking Area', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

