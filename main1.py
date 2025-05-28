# import cv2
# import easyocr
# from ultralytics import YOLO
# import re
# import os

# # === CONFIGURATION ===
# MODEL_PATH = 'best.pt'
# IMAGE_PATH = 'images/117.jpg'
# OUTPUT_DIR = 'output'
# CROPPED_DIR = os.path.join(OUTPUT_DIR, 'cropped')
# TEXT_FILE = os.path.join(OUTPUT_DIR, 'license_plate_text.txt')
# CONF_THRESHOLD = 0.3

# # Initialize EasyOCR reader
# reader = easyocr.Reader(['en'], gpu=False)

# # Load YOLO model
# model = YOLO(MODEL_PATH)

# # Create output folders
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(CROPPED_DIR, exist_ok=True)

# # Load image
# image = cv2.imread(IMAGE_PATH)
# if image is None:
#     raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# results = model.predict(source=IMAGE_PATH, save=False, conf=CONF_THRESHOLD)
# plate_texts = []

# # === TEXT CLEANER ===
# def clean_text(text):
#     return re.sub(r'[^A-Z0-9]', '', text.upper())

# # === DETECTION LOOP ===
# for r in results:
#     for i, box in enumerate(r.boxes):
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cropped = image[y1:y2, x1:x2]

#         # Save raw crop
#         crop_path = os.path.join(CROPPED_DIR, f"plate_{i+1}.jpg")
#         cv2.imwrite(crop_path, cropped)

#         # Convert to grayscale for EasyOCR (recommended)
#         gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

#         # Run EasyOCR
#         easyocr_results = reader.readtext(gray)
#         easy_text = ''
#         if easyocr_results:
#             easy_text = ' '.join([clean_text(res[1]) for res in easyocr_results])

#         plate_texts.append(easy_text)
#         print(f"[EasyOCR] Plate #{i+1}: {easy_text if easy_text else '(no text detected)'}")

#         # Annotate image
#         label = easy_text if easy_text else "License Plate"
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# # === SAVE FINAL RESULTS ===
# cv2.imwrite(os.path.join(OUTPUT_DIR, 'result.jpg'), image)
# with open(TEXT_FILE, 'w') as f:
#     for plate in plate_texts:
#         f.write(plate + '\n')

# print(f"\n[✅ DONE]")
# print(f"[📸] Annotated image saved to: {os.path.join(OUTPUT_DIR, 'result.jpg')}")
# print(f"[📄] Extracted plate text: {TEXT_FILE}")
# print(f"[📂] Cropped plates saved to: {CROPPED_DIR}")
import cv2
import pytesseract
import easyocr
from ultralytics import YOLO
import re
import os
import numpy as np

# === CONFIGURATION ===
MODEL_PATH = 'best.pt'
IMAGE_PATH = r'D:\RESEARCH\license plate detect\violations\crops\vehicle_310_id54.jpg'
OUTPUT_DIR = 'output'
CROPPED_DIR = os.path.join(OUTPUT_DIR, 'cropped')
PROCESSED_DIR = os.path.join(OUTPUT_DIR, 'processed')
TEXT_FILE = os.path.join(OUTPUT_DIR, 'license_plate_text.txt')
CONF_THRESHOLD = 0.3

# Set Tesseract executable path (Windows only)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=False)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Create output folders
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROPPED_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Load image
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

results = model.predict(source=IMAGE_PATH, save=False, conf=CONF_THRESHOLD)
plate_texts = []

# === TEXT CLEANER ===
def clean_text(text):
    return re.sub(r'[^A-Z0-9]', '', text.upper())

# === ROTATION CORRECTION ===
def deskew(image):
    coords = cv2.findNonZero(image)
    if coords is None:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# === DETECTION LOOP ===
for r in results:
    for i, box in enumerate(r.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped = image[y1:y2, x1:x2]

        # Save raw crop
        crop_path = os.path.join(CROPPED_DIR, f"plate_{i+1}.jpg")
        cv2.imwrite(crop_path, cropped)

        # === PREPROCESS FOR OCR ===
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 11, 3
        )

        rotated = deskew(thresh)
        proc_path = os.path.join(PROCESSED_DIR, f"processed_{i+1}.jpg")
        cv2.imwrite(proc_path, rotated)

        # === OCR STEP 1: EASYOCR ===
        easyocr_results = reader.readtext(gray)
        easy_text = ''
        if easyocr_results:
            easy_text = ' '.join([clean_text(res[1]) for res in easyocr_results])
        
        # === FALLBACK TO TESSERACT IF EASYOCR FAILED ===
        if not easy_text or len(easy_text) < 4:
            config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            tesseract_text = pytesseract.image_to_string(rotated, config=config).strip()
            tesseract_text = clean_text(tesseract_text.replace('\n', ' '))
            plate_texts.append(tesseract_text)
            print(f"[Tesseract] Plate #{i+1}: {tesseract_text if tesseract_text else '(no text detected)'}")
        else:
            plate_texts.append(easy_text)
            print(f"[EasyOCR] Plate #{i+1}: {easy_text}")

        # Annotate image
        label = plate_texts[-1] if plate_texts[-1] else "License Plate"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# === SAVE FINAL RESULTS ===
cv2.imwrite(os.path.join(OUTPUT_DIR, 'result.jpg'), image)
with open(TEXT_FILE, 'w') as f:
    for plate in plate_texts:
        f.write(plate + '\n')

print(f"\n[✅ DONE]")
print(f"[📸] Annotated image saved to: {os.path.join(OUTPUT_DIR, 'result.jpg')}")
print(f"[📄] Extracted plate text: {TEXT_FILE}")
print(f"[📂] Cropped plates saved to: {CROPPED_DIR}")
print(f"[📂] Preprocessed OCR inputs saved to: {PROCESSED_DIR}")
