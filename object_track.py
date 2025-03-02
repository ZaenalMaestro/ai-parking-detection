import cv2

from ultralytics import YOLO
import matplotlib.pyplot as plt
from ultralytics.utils.plotting import Annotator
from utils import read_license_plate

# Load the YOLO11 model
model = YOLO("yolo11n.pt")
plat_detector = YOLO('license_plate_detector.pt')
def detector(frame):
    results = plat_detector.predict(frame, show=False, imgsz=320, verbose=False)

    result = results[0]
    boxes = result.boxes
    text = None
    if boxes:
        for box in boxes:
            box_xyxy = box.xyxy.cpu().tolist()[0]

            plat_number = frame[int(box_xyxy[1]) : int(box_xyxy[3]), int(box_xyxy[0]) : int(box_xyxy[2])]
            plat_number = cv2.cvtColor(plat_number, cv2.COLOR_BGR2RGB)
            license_plate_crop_gray = cv2.cvtColor(plat_number, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
            text = read_license_plate(license_plate_crop_thresh)

            return text


def show_frame(frame):
    plt.imshow(frame)
    plt.show()

# Open the video file
video_path = "plat.mp4"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2)

vehicles = [2, 3, 5, 7]

plat_detected = {}

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame = cv2.resize(frame, (width, height))
    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, tracker='botsort.yaml', verbose=False, conf=0.6)
        boxes = results[0].boxes
        classes = results[0].names
        
        for box in boxes:
            track_id = -1
            if box.id is not None:
                track_id = int(box.id[0].item())

            class_id = box.cls[0].item()
            class_name = classes[class_id]
            if class_id not in vehicles:
                continue

            box_xyxy = box.xyxy.cpu().tolist()[0]

            vehicle_detected = frame[int(box_xyxy[1]) : int(box_xyxy[3]), int(box_xyxy[0]) : int(box_xyxy[2])]

            plat_number = ''
            if track_id != -1 and plat_detected.get(track_id) is None:
                plat_number = detector(vehicle_detected)
                plat_detected[track_id]= plat_number
            else:
                plat_number = plat_detected.get(track_id)


            annotator = Annotator(frame, line_width=2, example=class_name)
            annotator.box_label(box_xyxy, f'PLAT NUMBER: {plat_number}', (255, 0, 255), False)



        # Visualize the results on the frame
        # annotated_frame = results[0].plot()
        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()