from PIL import Image
import pytesseract
print(pytesseract.image_to_string(Image.open('tes.jpg')))