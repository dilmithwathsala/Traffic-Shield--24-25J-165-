import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, Point
import os
import pytesseract
import re
import easyocr

# === CONFIGURATION ===
YOLO_MODEL_PATH = 'yolov8n.pt'
PLATE_MODEL_PATH = 'best1.pt'
VIDEO_PATH = 'My Video2.mp4'
OUTPUT_VIDEO_PATH = 'output_video.avi'
CONF_THRESHOLD = 0.5
OUTPUT_RESOLUTION = (1000, 700)
PLATE_CONF_THRESHOLD = 0.3

# Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize EasyOCR reader
easyocr_reader = easyocr.Reader(['en'], gpu=False)

# === HSV RANGES FOR TRAFFIC LIGHT COLORS ===
red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 100, 100])
red_upper2 = np.array([180, 255, 255])
green_lower = np.array([40, 100, 100])
green_upper = np.array([80, 255, 255])
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([40, 255, 255])

# === SETUP ===
os.makedirs('violations/crops', exist_ok=True)
os.makedirs('violations/plates', exist_ok=True)

vehicle_model = YOLO(YOLO_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 20.0, OUTPUT_RESOLUTION)

zone_polygon = Polygon([(500, 200), (1000, 200), (1000, 720), (500, 720)])
vehicle_tracks = {}
violated_ids = set()
plate_texts = []
frame_count = 0

def preprocess_plate(plate_crop):
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    scale_factor = 3
    gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def extract_plate_text_tesseract(plate_crop):
    thresh = preprocess_plate(plate_crop)
    config = '--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(thresh, config=config).strip()
    clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
    return clean_text

def extract_plate_text_easyocr(plate_crop):
    thresh = preprocess_plate(plate_crop)
    results = easyocr_reader.readtext(thresh)
    text = ''
    if results:
        text = ''.join([res[1] for res in results]).upper()
        text = re.sub(r'[^A-Z0-9]', '', text)
    return text

def extract_plate_text_combined(plate_crop):
    tesseract_text = extract_plate_text_tesseract(plate_crop)
    easyocr_text = extract_plate_text_easyocr(plate_crop)
    # Choose longer result as more reliable
    if len(easyocr_text) > len(tesseract_text):
        print(f"[INFO] Using EasyOCR: {easyocr_text}")
        return easyocr_text
    else:
        print(f"[INFO] Using Tesseract: {tesseract_text}")
        return tesseract_text

def detect_traffic_light_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red = cv2.countNonZero(cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2))
    green = cv2.countNonZero(cv2.inRange(hsv, green_lower, green_upper))
    yellow = cv2.countNonZero(cv2.inRange(hsv, yellow_lower, yellow_upper))
    if red > green and red > yellow:
        return "Red"
    elif green > red and green > yellow:
        return "Green"
    elif yellow > red and yellow > green:
        return "Yellow"
    return "Unknown"

# === MAIN LOOP ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    traffic_light_color = "Unknown"
    results = vehicle_model.track(frame, persist=True, conf=CONF_THRESHOLD)

    cv2.polylines(frame, [np.array(zone_polygon.exterior.coords, np.int32)], True, (0, 0, 255), 2)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = vehicle_model.names[cls]
            track_id = int(box.id) if hasattr(box, "id") else None
            if conf < CONF_THRESHOLD:
                continue

            if label == "traffic light":
                roi = frame[y1:y1 + (y2 - y1) // 3, x1:x2]
                if roi.size > 0:
                    traffic_light_color = detect_traffic_light_color(roi)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"{label}: {traffic_light_color}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if label in ["car", "truck", "motorbike", "motorcycle", "bus", "bicycle"]:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                center = Point(cx, cy)
                prev = vehicle_tracks.get(track_id)
                vehicle_tracks[track_id] = (cx, cy)
                moving = np.linalg.norm(np.array((cx, cy)) - np.array(prev)) > 8 if prev else False

                if moving and track_id not in violated_ids and traffic_light_color == "Red" and zone_polygon.contains(center):
                    violated_ids.add(track_id)
                    crop_img = frame[y1:y2, x1:x2]
                    crop_path = f"violations/crops/vehicle_{frame_count}_id{track_id}.jpg"
                    cv2.imwrite(crop_path, crop_img)
                    cv2.imwrite(f"violations/frame_{frame_count}_id{track_id}.jpg", frame)

                    # === Number Plate Detection on Cropped Vehicle ===
                    plate_result = plate_model.predict(crop_img, save=False, conf=PLATE_CONF_THRESHOLD)
                    for i, plate_box in enumerate(plate_result[0].boxes):
                        px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                        plate_crop = crop_img[py1:py2, px1:px2]
                        plate_path = f"violations/plates/plate_{frame_count}_id{track_id}.jpg"
                        cv2.imwrite(plate_path, plate_crop)

                        plate_text = extract_plate_text_combined(plate_crop)
                        plate_texts.append(f"Vehicle {track_id} (Frame {frame_count}): {plate_text}")
                        print(f"[✅ Plate] Vehicle {track_id}: {plate_text}")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "VIOLATION", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    resized = cv2.resize(frame, OUTPUT_RESOLUTION)
    out.write(resized)
    cv2.imshow("Violation & Plate Detection", resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Save plate texts
with open('violations/plates.txt', 'w') as f:
    for line in plate_texts:
        f.write(line + '\n')

print(f"\n[✅ COMPLETED]")
print(f"[📸] Video saved: {OUTPUT_VIDEO_PATH}")
print(f"[📂] Cropped vehicles: violations/crops/")
print(f"[📂] Cropped plates: violations/plates/")
print(f"[📄] Plate text log: violations/plates.txt")
