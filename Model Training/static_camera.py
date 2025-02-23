import cv2
import numpy as np
from ultralytics import YOLO
import time

def load_yolo(model_path):
    return YOLO(model_path)

class_names = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]
vehicle_classes = {0, 1, 2, 3, 4}  # Only vehicle classes (excluding pedestrians)

class_colors = {
    0: (255, 0, 0),   # Car - Red
    1: (0, 255, 0),   # Truck - Green
    2: (0, 0, 255),   # Bus - Blue
    3: (255, 255, 0), # Motorcycle - Yellow
    4: (255, 0, 255), # Bicycle - Magenta
    5: (0, 255, 255)  # Person - Cyan
}

tracked_vehicles = {}  # {obj_id: first_seen_time} Track unique vehicles

TRAFFIC_SMOOTHING_WINDOW = 5  # Number of frames to average over
traffic_history = []          # List to store past traffic levels for smoothing
vehicle_arrival_times = []    # List to store timestamps of vehicle arrivals

last_traffic_update = time.time()

traffic_level = "Low"

# Separate counters for each class
class_counts = {name: 0 for name in class_names}

def classify_traffic(cars_per_second):
    if cars_per_second > 4.0:  # High traffic (adjust threshold as needed)
        return "High"
    else:
        return "Low"

def classify_pedestrian_difficulty(traffic_level):
    if traffic_level == "High":
        return "Tough Crossing"
    else:
        return "Easy Crossing"

def run_static_camera_detection(video_path, model_path, conf_threshold=0.3, iou_threshold=0.5):
    global tracked_vehicles, last_traffic_update, traffic_level, traffic_history, class_counts, vehicle_arrival_times
    model = load_yolo(model_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Error: Unable to open video file.")
        return
    
    frame_count = 0

    roi_x1 = 0
    roi_y1 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.4)  
    roi_x2 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    roi_y2 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]
        
        results = model.track(frame, persist=True, conf=conf_threshold, iou=iou_threshold, imgsz=1280)
        
        if results[0].boxes is None:
            frame_count += 1
            if traffic_history: # ensure the history is not empty
                smoothed_traffic = max(set(traffic_history), key=traffic_history.count)
                traffic_level = smoothed_traffic  # Update the global traffic_level

            # Display traffic and pedestrian crossing info in the top-left
            cv2.putText(frame, f"Traffic: {traffic_level}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Pedestrian: {classify_pedestrian_difficulty(traffic_level)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Display class counts in the top-right
            y_offset = 40
            for i, name in enumerate(class_names):
                text = f"{name}: {class_counts[name]}"
                cv2.putText(frame, text, (frame_width - 200, y_offset + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
            
            cv2.imshow("Static Camera Traffic Monitoring", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue  # No detections in this frame
        

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            obj_id = int(box.id[0].item())

            box_area = (x2 - x1) * (y2 - y1) # calculate area of box
            
            if cls >= len(class_names) or conf < conf_threshold:
                continue
            
            if cls in vehicle_classes:
                if roi_x1 < x1 and roi_y1 < y1 and roi_x2 > x2 and roi_y2 > y2:
                    if obj_id not in tracked_vehicles:
                        tracked_vehicles[obj_id] = time.time() # track the vehicle ids and their time in the frame
                        vehicle_arrival_times.append(time.time()) # Record the arrival time
                        
                        # Min area check here
                        if box_area > 5000:
                            class_counts[class_names[cls]] += 1  # Increment the count for this class


                    color = class_colors.get(cls, (255, 255, 255))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{class_names[cls]} {conf:.2f} ID: {obj_id}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

        current_time = time.time()
        time_window = 1

        #Calculate the rate
        arrival_times_in_window = [t for t in vehicle_arrival_times if current_time - t <= time_window] # count vehicle arrivals in the last time window
        cars_per_second = len(arrival_times_in_window) / time_window # Number of cars divided by time_window

        traffic_level = classify_traffic(cars_per_second)
        traffic_history.append(traffic_level)

        if len(traffic_history) > TRAFFIC_SMOOTHING_WINDOW:
            traffic_history.pop(0)

        smoothed_traffic = max(set(traffic_history), key=traffic_history.count)
        traffic_level = smoothed_traffic

        # Display traffic and pedestrian crossing info in the top-left
        cv2.putText(frame, f"Traffic: {traffic_level}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Pedestrian: {classify_pedestrian_difficulty(traffic_level)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Display class counts in the top-right
        y_offset = 40
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        for i, name in enumerate(class_names):
            text = f"{name}: {class_counts[name]}"
            cv2.putText(frame, text, (frame_width - 200, y_offset + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


        cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

        cv2.imshow("Static Camera Traffic Monitoring", frame)
        
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

input_video = r"C:\Users\garg1\OneDrive\Desktop\PS1\3\british_highway_traffic.mp4" # Ensure there is a video at that this path
model_path = r"C:\Users\garg1\OneDrive\Desktop\KrackHacK-akatsuki\Website-rendering\best.pt" # Ensure there is the trained model at this path

run_static_camera_detection(input_video, model_path)