import cv2
from ultralytics import YOLO

model = YOLO("best(b1).pt") 

# Open the video
cap = cv2.VideoCapture('videos/video 5.mp4')
count_line_position = 950
line_length = 1920  
offset = 6
counter = 0

counted_vehicles = {}
crossing_vehicles = {}

vehicle_classes = ['truck', 'auto rickshaw', 'bicycle', 'car', 'rickshaw',
                   'motorbike', 'bus', 'pickup', 'taxi', 'suv']

vehicle_type_counter = {vehicle: 0 for vehicle in vehicle_classes}

def is_crossing_line(y, line_y, offset):
    return line_y - offset < y < line_y + offset

playback_delay = 50 

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)[0]
    detections = results.boxes.data  

    cv2.line(frame, (25, count_line_position), (line_length, count_line_position), (255, 127, 0), 3)

    current_vehicles = set()

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        class_name = model.names[int(cls_id)]

        print(f"Detected: {class_name} (Confidence: {conf:.2f})")

        if conf > 0.4 and class_name in vehicle_classes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            grid_x = cx // 50  
            vehicle_id = f"{class_name}_{grid_x}"

            if is_crossing_line(cy, count_line_position, offset):
                if vehicle_id not in counted_vehicles and vehicle_id not in crossing_vehicles:
                    counter += 1
                    counted_vehicles[vehicle_id] = True
                    crossing_vehicles[vehicle_id] = True
                    vehicle_type_counter[class_name] += 1  
                    cv2.line(frame, (25, count_line_position), (line_length, count_line_position), (0, 127, 255), 3)
            else:
                if vehicle_id in crossing_vehicles:
                    del crossing_vehicles[vehicle_id]

            current_vehicles.add(vehicle_id)

            color = (0, 255, 0) if class_name == "car" else \
                    (255, 0, 0) if class_name == "motorbike" else \
                    (0, 0, 255) if class_name == "truck" else \
                    (0, 255, 255)  

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, class_name.upper(), (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

    counted_vehicles = {k: v for k, v in counted_vehicles.items() if k in current_vehicles}

    cv2.putText(frame, f"Vehicle Count: {counter}", (450, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    y_offset = 110
    for idx, (veh, count_val) in enumerate(vehicle_type_counter.items()):
        cv2.putText(frame, f"{veh.title()}: {count_val}", (450, y_offset + idx*30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f"Playback: Slow ({playback_delay}ms delay)", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Vehicle Detection", frame)

    if cv2.waitKey(playback_delay) & 0xFF == 13:  
        break

cap.release()
cv2.destroyAllWindows()
