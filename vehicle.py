import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, Point
import os

# === CONFIGURATION ===
MODEL_PATH = 'yolov8n.pt'
VIDEO_PATH = 'My Video2.mp4'
OUTPUT_VIDEO_PATH = 'output_video.avi'
OUTPUT_RESOLUTION = (1000, 700)
CONF_THRESHOLD = 0.5

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
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 20.0, OUTPUT_RESOLUTION)

os.makedirs('violations', exist_ok=True)
os.makedirs('violations/crops', exist_ok=True)
vehicle_tracks = {}
violated_ids = set()
frame_count = 0

# === Define Detection Zone ===
zone_polygon = Polygon([(500, 200), (1000, 200), (1000, 720), (500, 720)])

# === Traffic Light Detection Function ===
def detect_traffic_light_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    red = cv2.countNonZero(red_mask)
    green = cv2.countNonZero(green_mask)
    yellow = cv2.countNonZero(yellow_mask)

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

    # Run YOLO tracking
    results = model.track(frame, persist=True, conf=CONF_THRESHOLD)
    frame_h, frame_w = frame.shape[:2]

    # Draw violation zone
    pts = np.array(zone_polygon.exterior.coords, np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            track_id = int(box.id) if hasattr(box, "id") else None

            if conf < CONF_THRESHOLD:
                continue

            # === Traffic light detection ===
            if label == "traffic light":
                roi = frame[y1:y1 + (y2 - y1) // 3, x1:x2]
                if roi.size > 0:
                    traffic_light_color = detect_traffic_light_color(roi)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"{label}: {traffic_light_color}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # === Vehicle detection ===
            if label in ["car", "truck", "motorbike", "motorcycle", "bus", "bicycle"]:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                center = Point(cx, cy)

                # Update tracking
                prev = vehicle_tracks.get(track_id)
                vehicle_tracks[track_id] = (cx, cy)

                # Check movement
                moving = False
                if prev:
                    speed = np.linalg.norm(np.array((cx, cy)) - np.array(prev))
                    moving = speed > 8

                # === Violation Check (only if moving) ===
                if moving:
                    if (track_id not in violated_ids and
                        traffic_light_color == "Red" and
                        zone_polygon.contains(center)):

                        # Mark violation
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, "VIOLATION", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        violated_ids.add(track_id)

                        # Save full frame
                        cv2.imwrite(f"violations/frame_{frame_count}_id{track_id}.jpg", frame)

                        # Save cropped vehicle
                        cropped = frame[y1:y2, x1:x2]
                        crop_path = f"violations/crops/vehicle_{frame_count}_id{track_id}.jpg"
                        cv2.imwrite(crop_path, cropped)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    # Stationary
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                # Labels
                cv2.putText(frame, f"{label} ID:{track_id}", (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"{conf:.2f}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    # Show & save
    resized = cv2.resize(frame, OUTPUT_RESOLUTION)
    out.write(resized)
    cv2.imshow("Red Light Violation Detection", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === CLEANUP ===
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[✅] Done! Output saved to: {OUTPUT_VIDEO_PATH}")
