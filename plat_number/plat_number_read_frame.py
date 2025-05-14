import cv2
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
from plat_number.plat_number_normalization import read_license_plate

model = YOLO('license_plate_detector.pt')
names = model.names


reader = easyocr.Reader(['en'])
img_path = 'plat_number/plat number frame.png'
image = cv2.imread(img_path)

def show_frame(frame):
    plt.imshow(frame)
    plt.show()

def plat_number(frame):
    results = model.predict(frame, show=False, verbose=False)
    result = results[0]
    boxes = result.boxes
    plat_number_result = None


    if boxes:
        for box in boxes:
            box_xyxy = box.xyxy.cpu().tolist()[0]
            x1 = int(box_xyxy[0])
            y1 = int(box_xyxy[1])
            x2 = int(box_xyxy[2])
            y2 = int(box_xyxy[3])
            

            plat_number = frame[y1 : y2, x1 : x2]
            plat_number = cv2.cvtColor(plat_number, cv2.COLOR_BGR2RGB)
            plat_number_gray = cv2.cvtColor(plat_number, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(plat_number_gray, 120, 255, cv2.THRESH_BINARY_INV)
            plat_number_text, score = read_license_plate(license_plate_crop_thresh)
            plat_number_result = plat_number_text

        return plat_number_result

    return None
        




# Check if the image was loaded successfully
if image is None:
    print(f"Could not open or find the image: {img_path}")
else:
    # You can now work with the 'image' variable, which is a NumPy array
    print(f"Image shape: {image.shape}")  # Print the dimensions of the image
    result = plat_number(image)
    print('plat number:', result)
    # To display the image (optional):
    cv2.imshow('Image', image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close all display windows















# def show_frame(frame):
#     plt.imshow(frame)
#     plt.show()
    
# plat_number_analytics = {}

# ret, frame = cap.read()
# # frame = cv2.resize(frame, (lebar, tinggi))
# results = model.track(frame, persist=True, tracker='botsort.yaml', verbose=False)
# result = results[0]
# boxes = result.boxes


# if boxes:
#     for box in boxes:
#         box_xyxy = box.xyxy.cpu().tolist()[0]
#         x1 = int(box_xyxy[0])
#         y1 = int(box_xyxy[1])
#         x2 = int(box_xyxy[2])
#         y2 = int(box_xyxy[3])

#         track_id = -1
#         if box.id is not None:
#             track_id = int(box.id[0].item())
        

#         plat_number = frame[y1 : y2, x1 : x2]
#         show_frame(plat_number)
#         # plat_number = cv2.cvtColor(plat_number, cv2.COLOR_BGR2RGB)
#         # plat_number_gray = cv2.cvtColor(plat_number, cv2.COLOR_BGR2GRAY)
#         # _, license_plate_crop_thresh = cv2.threshold(plat_number_gray, 80, 255, cv2.THRESH_BINARY_INV)
#         # plat_number_text, score = read_license_plate(license_plate_crop_thresh)


#         # if (len(plat_number_analytics) > 0) and \
#         #    (plat_number_analytics.get(track_id) is not None) and \
#         #    (float(score) < plat_number_analytics[track_id]['confidence']):
#         #     plat_number_text = plat_number_analytics[track_id]['plat_number']
#         #     score = plat_number_analytics[track_id]['confidence']

#         # plat_number_analytics.update({track_id:
#         #     {
#         #         'plat_number': plat_number_text,
#         #         'confidence': score
#         #     }
#         # })
#         # annotator = Annotator(frame, line_width=2, example='names')
#         # annotator.box_label(box_xyxy, plat_number_text, (255, 100, 255), False)
# # cv2.waitKey(0)
# cv2.imshow('PLAT NUMBER DETECTION', frame)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break


# cap.release()
# cv2.destroyAllWindows()

