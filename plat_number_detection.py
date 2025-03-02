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
from utils import read_license_plate


model = YOLO('license_plate_detector.pt')
names = model.names

source = 'plat.mp4'
source1 = 'plat_number_video.mp4'
source2 = 'anpr-demo-video.mp4'
reader = easyocr.Reader(['en'])

cap = cv2.VideoCapture(source2)
lebar = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
tinggi = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

def show_frame(frame):
    plt.imshow(frame)
    plt.show()
    
plat_number_analytics = {}

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (lebar, tinggi))
    results = model.track(frame, persist=True, tracker='botsort.yaml', verbose=False)
    result = results[0]
    boxes = result.boxes


    if boxes:
        for box in boxes:
            box_xyxy = box.xyxy.cpu().tolist()[0]
            x1 = int(box_xyxy[0])
            y1 = int(box_xyxy[1])
            x2 = int(box_xyxy[2])
            y2 = int(box_xyxy[3])

            track_id = -1
            if box.id is not None:
                track_id = int(box.id[0].item())
            

            plat_number = frame[y1 : y2, x1 : x2]
            plat_number = cv2.cvtColor(plat_number, cv2.COLOR_BGR2RGB)
            plat_number_gray = cv2.cvtColor(plat_number, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(plat_number_gray, 80, 255, cv2.THRESH_BINARY_INV)
            plat_number_text, score = read_license_plate(license_plate_crop_thresh)


            if (len(plat_number_analytics) > 0) and \
               (plat_number_analytics.get(track_id) is not None) and \
               (float(score) < plat_number_analytics[track_id]['confidence']):
                plat_number_text = plat_number_analytics[track_id]['plat_number']
                score = plat_number_analytics[track_id]['confidence']

            plat_number_analytics.update({track_id:
                {
                    'plat_number': plat_number_text,
                    'confidence': score
                }
            })
            annotator = Annotator(frame, line_width=2, example='names')
            annotator.box_label(box_xyxy, plat_number_text, (255, 100, 255), False)

    cv2.imshow('PLAT NUMBER DETECTION', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

