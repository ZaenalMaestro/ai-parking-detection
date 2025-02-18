import cv2
import numpy as np

# Membuat citra kosong
img = np.zeros((512, 512, 3), np.uint8)
def checkMousePosition(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        print(f'x:{x} | y:{y}')

# [x, y]
illegal_area = np.array([[60, 94], [272, 153], [273, 275], [46, 222]], np.int32)

# cv2.polylines(img, [illegal_area], True, (0, 255, 0), 1)
# cv2.namedWindow('Parking Area')
# cv2.setMouseCallback('Parking Area', checkMousePosition)
# # Menampilkan citra
# cv2.imshow('Parking Area', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cap = cv2.VideoCapture('illegal-parking.mp4')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.namedWindow('Parking Area')
    cv2.setMouseCallback('Parking Area', checkMousePosition)

    cv2.polylines(frame, [illegal_area], True, (0, 0, 255), 3)
    cv2.namedWindow('Parking Area')
    cv2.imshow('Parking Area', frame)
    # cv2.waitKey(0)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

