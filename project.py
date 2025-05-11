import cv2
from ultralytics import YOLO

# Load your custom-trained YOLOv8 model
model = YOLO("best(b1).pt")  # <-- path to uploaded model

# Open the video
cap = cv2.VideoCapture('./videos/Video 4.mp4')
count_line_position = 600
offset = 6
counter = 0

#drone
#vehicle_classes = ['Auto-rickshaw', 'Bicycle', 'Bus', 'Car', 'Cycle-rickshaw', 'E-rickshaw', 'Motorcycle', 'Pedestrian', 'Tractor-trolley', 'Truck']


# best model
vehicle_classes = [ 'truck', 'auto rickshaw', 'bicycle', 'car', 'rickshaw', 'motorbike', 'bus', 'bicycle', "pickup", 'taxi', 'suv']  # Adjust as per your training data.yaml
# - ambulance
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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 detection
    results = model(frame)[0]
    detections = results.boxes.data  # [x1, y1, x2, y2, confidence, class]

    # Draw the counting line
    cv2.line(frame, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        class_name = model.names[int(cls_id)]

        print(f"Detected: {class_name} (Confidence: {conf:.2f})")

        if conf > 0.4 and class_name in vehicle_classes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if is_crossing_line(cy, count_line_position, offset):
                counter += 1
                cv2.line(frame, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)

            color = (0, 255, 0) if class_name == "car" else \
                    (255, 0, 0) if class_name == "motorcycle" else \
                    (0, 0, 255) if class_name == "truck" else \
                    (0, 255, 255)  # default color for others

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, class_name.upper(), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

    cv2.putText(frame, f"Vehicle Count: {counter}", (450, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    cv2.imshow("Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == 13:
        break

cap.release()
cv2.destroyAllWindows()
