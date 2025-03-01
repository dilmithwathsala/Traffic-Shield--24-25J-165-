import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov5s.pt')  # Replace with your YOLOv5 model

# Open video file
cap = cv2.VideoCapture("sliit1.mp4")

# Define the desired output video resolution
output_width = 640
output_height = 480

# Create a VideoWriter object to save the output video with the new resolution
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec if needed
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (output_width, output_height))

# List to track pedestrians and their positions
pedestrians_crossing = []
vehicle_positions = []  # This will track vehicle positions (bounding box coordinates)
vehicle_velocities = {}  # A dictionary to track the velocity of vehicles

# Traffic light color ranges in HSV
# Red traffic light ranges (two ranges for better detection)
red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([170, 100, 100])
red_upper2 = np.array([180, 255, 255])

# Green traffic light range
green_lower = np.array([40, 100, 100])
green_upper = np.array([80, 255, 255])

# Yellow traffic light range
yellow_lower = np.array([20, 100, 100])
yellow_upper = np.array([40, 255, 255])

# Function to detect traffic light color
def detect_traffic_light_color(roi):
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define the masks for red, green, and yellow traffic lights
    red_mask = cv2.inRange(hsv_roi, red_lower1, red_upper1) | cv2.inRange(hsv_roi, red_lower2, red_upper2)
    green_mask = cv2.inRange(hsv_roi, green_lower, green_upper)
    yellow_mask = cv2.inRange(hsv_roi, yellow_lower, yellow_upper)
    
    # Count non-zero pixels in each mask to determine the dominant color
    red_count = cv2.countNonZero(red_mask)
    green_count = cv2.countNonZero(green_mask)
    yellow_count = cv2.countNonZero(yellow_mask)
    
    if red_count > green_count and red_count > yellow_count:
        return "Red"
    elif green_count > red_count and green_count > yellow_count:
        return "Green"
    elif yellow_count > red_count and yellow_count > green_count:
        return "Yellow"
    else:
        return "Unknown"

# Function to calculate vehicle movement
def is_moving(curr_pos, prev_pos):
    # Compare the current and previous positions of the vehicle
    if curr_pos is not None and prev_pos is not None:
        x1, y1, x2, y2 = curr_pos
        prev_x1, prev_y1, prev_x2, prev_y2 = prev_pos
        movement_threshold = 10  # Minimum movement in pixels to consider it as moving
        # Check if the vehicle moved significantly
        if abs(x1 - prev_x1) > movement_threshold or abs(y1 - prev_y1) > movement_threshold:
            return True
    return False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)
    traffic_light_color = "Unknown"
    
    # Define the position of the red line (vertical line)
    red_line_x = 600  # Change this to adjust the line position

    # Draw the red line on the frame
    cv2.line(frame, (red_line_x, 0), (red_line_x, frame.shape[0]), (0, 0, 255), 2)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.tolist()[0])
            conf = box.conf.item()
            cls = int(box.cls.item())
            label = model.names[cls]

            # Detect traffic lights
            if label == "traffic light" and conf > 0.5:
                roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                if roi.size > 0:
                    traffic_light_color = detect_traffic_light_color(roi)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"{label} - {traffic_light_color}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Detect pedestrians
            if label == "person" and conf > 0.5:
                person_bottom_y = y2
                # Track pedestrians crossing the road
                if (x1, y1, x2, y2) not in pedestrians_crossing:
                    pedestrians_crossing.append((x1, y1, x2, y2))
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Detect vehicles
            if label in ["car", "truck", "motorbike", "bus"] and conf > 0.5:
                vehicle_bottom_y = y2
                
                # Track vehicles' previous positions
                prev_pos = vehicle_velocities.get((x1, y1, x2, y2), None)
                is_vehicle_moving = is_moving((x1, y1, x2, y2), prev_pos)
                
                # Check if a vehicle is crossing during red light and is moving
                if traffic_light_color == "Red" and is_vehicle_moving:
                    # Check if the vehicle has crossed the red line
                    if x2 > red_line_x:  # Vehicle crosses the red line
                        # Highlight the vehicle in red to indicate violation
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Vehicle Violation!", (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        # Store vehicle position for future reference if needed
                        vehicle_positions.append((x1, y1, x2, y2))

                # Update vehicle's previous position for tracking
                vehicle_velocities[(x1, y1, x2, y2)] = (x1, y1, x2, y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Resize the frame to the desired resolution
    resized_frame = cv2.resize(frame, (1000, 1000))

    # Write the resized frame to the output video
    out.write(resized_frame)

    # Display the resized frame
    cv2.imshow('Pedestrian and Vehicle Crossing Violation Detection', resized_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video writer and reader
cap.release()
out.release()
cv2.destroyAllWindows()
