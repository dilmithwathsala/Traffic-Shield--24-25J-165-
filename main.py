import cv2
from ultralytics import YOLO
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import os

# === CONFIG ===
MODEL_PATH = 'best.pt'
IMAGE_PATH = 'images/117.jpg'
OUTPUT_DIR = 'output'
CONF_THRESHOLD = 0.3
TEXT_FILE = os.path.join(OUTPUT_DIR, 'license_plate_text.txt')

# Optional: set tesseract path (only for Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# === SETUP ===
model = YOLO(MODEL_PATH)
os.makedirs(OUTPUT_DIR, exist_ok=True)
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not find image at {IMAGE_PATH}")

results = model.predict(source=IMAGE_PATH, save=False, conf=CONF_THRESHOLD)

plate_texts = []

# Process detections
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"License Plate"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # === CROP & OCR ===
        cropped = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config='--psm 7')  # psm 7: treat as single line
        text = text.strip()
        plate_texts.append(text)
        print(f"[INFO] Detected Text: {text}")

# Save image with box
cv2.imwrite(os.path.join(OUTPUT_DIR, 'result.jpg'), image)

# Save text to file
with open(TEXT_FILE, 'w') as f:
    for plate in plate_texts:
        f.write(plate + '\n')

print(f"[INFO] Saved extracted text to: {TEXT_FILE}")
