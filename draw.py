import cv2
import numpy as np

# Membuat citra kosong
img = np.zeros((512, 512, 3), np.uint8)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

# Titik-titik yang membentuk poligon
# [x, y]
pts = np.array([[107, 166], [357, 166], [434, 421], [86, 421]], np.int32)

# Menggambar poligon
cv2.polylines(img, [pts], True, (0, 255, 0), 1)
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
# Menampilkan citra
cv2.imshow('RGB', img)
cv2.waitKey(0)
cv2.destroyAllWindows()