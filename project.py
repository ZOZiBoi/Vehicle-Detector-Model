import cv2
from ultralytics import YOLO

# Load your custom-trained YOLOv8 model
model = YOLO("best(b1).pt")  # <-- path to uploaded model

# Open the video
cap = cv2.VideoCapture('videos/video 10.mp4')
count_line_position = 950
line_length = 1920  
offset = 6
counter = 0

# Dictionary to track vehicles that have been counted
counted_vehicles = {}
# Dictionary to track vehicles that are currently crossing the line
crossing_vehicles = {}

# drone
#vehicle_classes = ['Auto-rickshaw', 'Bicycle', 'Bus', 'Car', 'Cycle-rickshaw', 'E-rickshaw', 'Motorcycle', 'Pedestrian', 'Tractor-trolley', 'Truck']


# best model
vehicle_classes = [ 'truck', 'auto rickshaw', 'bicycle', 'car', 'rickshaw', 'motorbike', 'bus', 'bicycle', "pickup", 'taxi', 'suv']  # Adjust as per your training data.yaml
# - army vehicle
# - auto rickshaw
# - bicycle
# - bus
# - car
# - garbagevan
# - human hauler
# - minibus
# - minivan
# - motorbike
# - pickup
# - policecar
# - rickshaw
# - scooter
# - suv
# - taxi
# - three wheelers -CNG-
# - truck
# - van
# - wheelbarrow
def is_crossing_line(y, line_y, offset):
    return line_y - offset < y < line_y + offset

# Add playback speed control
playback_delay = 50  # milliseconds between frames (higher = slower)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)[0]
    detections = results.boxes.data  # [x1, y1, x2, y2, confidence, class]

    # Draw the counting line
    cv2.line(frame, (25, count_line_position), (line_length, count_line_position), (255, 127, 0), 3)

    # Track current frame vehicles
    current_vehicles = set()

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        class_name = model.names[int(cls_id)]

        print(f"Detected: {class_name} (Confidence: {conf:.2f})")

        if conf > 0.4 and class_name in vehicle_classes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Create a unique identifier for the vehicle based on its class and approximate position
            # Using a grid-based approach to make the ID more stable
            grid_x = cx // 50  # Divide the frame into 50-pixel wide grids
            vehicle_id = f"{class_name}_{grid_x}"

            if is_crossing_line(cy, count_line_position, offset):
                if vehicle_id not in counted_vehicles and vehicle_id not in crossing_vehicles:
                    # Vehicle is crossing the line for the first time
                    counter += 1
                    counted_vehicles[vehicle_id] = True
                    crossing_vehicles[vehicle_id] = True
                    cv2.line(frame, (25, count_line_position), (line_length, count_line_position), (0, 127, 255), 3)
            else:
                # If vehicle is not crossing the line, remove it from crossing_vehicles
                if vehicle_id in crossing_vehicles:
                    del crossing_vehicles[vehicle_id]
            
            current_vehicles.add(vehicle_id)

            color = (0, 255, 0) if class_name == "car" else \
                    (255, 0, 0) if class_name == "motorcycle" else \
                    (0, 0, 255) if class_name == "truck" else \
                    (0, 255, 255)  # default color for others

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, class_name.upper(), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

    # Remove vehicles that are no longer in frame from counted_vehicles
    counted_vehicles = {k: v for k, v in counted_vehicles.items() if k in current_vehicles}

    cv2.putText(frame, f"Vehicle Count: {counter}", (450, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # Display playback speed info
    cv2.putText(frame, f"Playback: Slow ({playback_delay}ms delay)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
    cv2.imshow("Vehicle Detection", frame)

    # Use the playback delay value instead of 1ms
    if cv2.waitKey(playback_delay) & 0xFF == 13:
        break

cap.release()
cv2.destroyAllWindows()
