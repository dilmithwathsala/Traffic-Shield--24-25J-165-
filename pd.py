import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, Point
import os
import pytesseract

# === CONFIGURATION ===
YOLO_MODEL_PATH = 'yolov8n.pt'
PLATE_MODEL_PATH = 'best.pt'
VIDEO_PATH = 'pedestrian.mp4'
OUTPUT_VIDEO_PATH = 'output_video.avi'
CONF_THRESHOLD = 0.5
OUTPUT_RESOLUTION = (1000, 700)
PLATE_CONF_THRESHOLD = 0.3

# Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initial pedestrian crossing polygon points (mutable)
crossing_points = [
    [500, 460],  # bottom-left
    [1300, 450],  # bottom-right
    [1400, 680],  # near bottom-right
    [400, 680]   # near bottom-left
]

# --- BEGIN PEDESTRIAN CROSSING ZONE POSITION ADJUSTMENT ---
# Adjust these values to move the polygon
offset_x = 50   # Positive: move right, Negative: move left
offset_y = 400   # Positive: move down, Negative: move up

# Apply offset to polygon points
crossing_points = [[x + offset_x, y + offset_y] for x, y in crossing_points]
# --- END PEDESTRIAN CROSSING ZONE POSITION ADJUSTMENT ---

# Setup folders
os.makedirs('violations/crops', exist_ok=True)
os.makedirs('violations/plates', exist_ok=True)
os.makedirs('violations/pedestrian_crossing_violations', exist_ok=True)  # New folder for pedestrian crossing violations

# Load models
vehicle_model = YOLO(YOLO_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 20.0, OUTPUT_RESOLUTION)

# Tracking and state
vehicle_tracks = {}
violated_ids = set()
plate_texts = []
frame_count = 0

# Parameters for draggable points
dragging_point = None
radius = 15  # radius for draggable corner circles

def extract_plate_text(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return ''
    try:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(
            gray,
            config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ).strip()
        print(f"[INFO] Detected Plate Text: {text}")
        return text
    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")
        return ''

def point_near(pt1, pt2, dist=15):
    return np.linalg.norm(np.array(pt1) - np.array(pt2)) < dist

def mouse_callback(event, x, y, flags, param):
    global dragging_point, crossing_points

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check if click near any corner point to start dragging
        for i, pt in enumerate(crossing_points):
            if point_near((x, y), pt, dist=radius):
                dragging_point = i
                break

    elif event == cv2.EVENT_LBUTTONUP:
        dragging_point = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_point is not None:
            # Update the position of the dragged corner
            crossing_points[dragging_point] = [x, y]

cv2.namedWindow("Pedestrian Violation & Plate Detection")
cv2.setMouseCallback("Pedestrian Violation & Plate Detection", mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    crossing_zone = Polygon(crossing_points)

    results = vehicle_model.track(frame, persist=True, conf=CONF_THRESHOLD)

    # Draw pedestrian crossing polygon and corner circles
    pts = np.array(crossing_points, np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 0), thickness=3)
    for pt in crossing_points:
        cv2.circle(frame, tuple(map(int, pt)), radius, (0, 255, 255), -1)

    cv2.putText(frame, "Pedestrian Crossing Zone",
                (int(crossing_zone.bounds[0]), int(crossing_zone.bounds[1]) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    pedestrian_inside = set()

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = vehicle_model.names[cls]
            track_id = int(box.id) if hasattr(box, "id") else None
            if conf < CONF_THRESHOLD:
                continue

            center = Point((x1 + x2) // 2, (y1 + y2) // 2)

            prev_pos = vehicle_tracks.get(track_id)
            vehicle_tracks[track_id] = (center.x, center.y)

            if label == 'person' and crossing_zone.contains(center):
                pedestrian_inside.add(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"Pedestrian ID:{track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                continue

            if label in ["car", "truck", "motorbike", "motorcycle", "bus", "bicycle"]:
                moving = False
                if prev_pos is not None:
                    moving = np.linalg.norm(np.array((center.x, center.y)) - np.array(prev_pos)) > 8

                if moving and track_id not in violated_ids and crossing_zone.contains(center) and len(pedestrian_inside) > 0:
                    violated_ids.add(track_id)

                    # Save cropped vehicle image in pedestrian crossing violations folder
                    violation_crop_path = f"violations/pedestrian_crossing_violations/vehicle_{frame_count}_id{track_id}.jpg"
                    cv2.imwrite(violation_crop_path, frame[y1:y2, x1:x2])

                    # Also save in existing crops folder
                    crop_img = frame[y1:y2, x1:x2]
                    crop_path = f"violations/crops/vehicle_{frame_count}_id{track_id}.jpg"
                    cv2.imwrite(crop_path, crop_img)
                    cv2.imwrite(f"violations/frame_{frame_count}_id{track_id}.jpg", frame)

                    # Detect number plate on cropped vehicle
                    plate_result = plate_model.predict(crop_img, save=False, conf=PLATE_CONF_THRESHOLD)
                    for i, plate_box in enumerate(plate_result[0].boxes):
                        px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                        plate_crop = crop_img[py1:py2, px1:px2]
                        plate_path = f"violations/plates/plate_{frame_count}_id{track_id}.jpg"
                        cv2.imwrite(plate_path, plate_crop)

                        plate_text = extract_plate_text(plate_crop)
                        plate_texts.append(f"Vehicle {track_id} (Frame {frame_count}): {plate_text}")
                        print(f"[✅ Plate] Vehicle {track_id}: {plate_text}")

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "PEDESTRIAN VIOLATION", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"{label}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    resized = cv2.resize(frame, OUTPUT_RESOLUTION)
    out.write(resized)
    cv2.imshow("Pedestrian Violation & Plate Detection", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

with open('violations/plates.txt', 'w') as f:
    for line in plate_texts:
        f.write(line + '\n')

print(f"\n[✅ COMPLETED]")
print(f"[📸] Video saved: {OUTPUT_VIDEO_PATH}")
print(f"[📂] Cropped vehicles: violations/crops/")
print(f"[📂] Cropped plates: violations/plates/")
print(f"[📂] Pedestrian Crossing Violations: violations/pedestrian_crossing_violations/")
print(f"[📄] Plate text log: violations/plates.txt")
print(f"[🚦] Total violations detected: {len(violated_ids)}")
