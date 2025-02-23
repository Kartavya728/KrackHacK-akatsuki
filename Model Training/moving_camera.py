import cv2
import numpy as np
from ultralytics import YOLO

# {0: 'car', 1: 'truck", 2: "bus", 3: "motorcycle", 4: "bicycle", 5: "person", 6: "rider", 7: "traffic light", 8: "traffic sign", "lane", "drivable area"}
class_colors = {
    0: (255, 0, 0),   # car - Red
    1: (0, 0, 255),   # truck - Blue
    2: (0, 255, 0),   # bus - Green
    3: (0, 0, 255),   # Motorcycle - Blue
    4: (0, 0, 255),   # bicycle - Blue
    5: (255, 0, 0),   # person - Red
    6: (255, 255, 255), # rider - White
    7: (255, 255, 0), # traffic-light - Yellow
    8: (0, 0, 255),   # traffic-sign - Blue
    9: (255, 0, 0),   # lane - Red
    10: (0, 255, 0)   # drivable area - Green
}

def auto_rotate(frame):
    h, w = frame.shape[:2]
    return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) if h > w else frame

def detect_lane(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    mask = np.zeros_like(edges)
    h, w = frame.shape[:2]
    mask[int(h * 0.6):, :] = 255  # Focus on bottom half
    edges = cv2.bitwise_and(edges, mask)
    return edges

def get_lane_mask(edges, frame_shape):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    yellow_edges = np.where(edges > 0)

    if len(yellow_edges[0]) == 0 or yellow_edges[1].size == 0:
        return mask

    left_x, right_x = np.min(yellow_edges[1]), np.max(yellow_edges[1])
    bottom_y, top_y = np.max(yellow_edges[0]), np.min(yellow_edges[0])

    points = np.array([[left_x, bottom_y], [left_x, top_y], [right_x, top_y], [right_x, bottom_y]], np.int32)
    cv2.fillPoly(mask, [points], 255)
    return mask

def get_drivable_area_mask(frame, detections, lane_mask, edges):
    """Generates a single, centered drivable area mask using detected drivable areas and lane edges."""
    h, w = frame.shape[:2]
    drivable_area_mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)  # Single channel mask

    best_drivable_area = None
    max_area = 0
    best_y2 = 0

    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0].item())
        class_names = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "rider", "traffic light", "traffic sign", "lane", "drivable area"]

        if cls < len(class_names) and class_names[cls] == "drivable area":
            # Calculate area of the bounding box
            area = (x2 - x1) * (y2 - y1)

            # Store the largest drivable area that is closer (larger y2)
            if area > max_area or y2 > best_y2:
                max_area = area
                best_drivable_area = (x1, y1, x2, y2)
                best_y2 = y2  # Store y2 for proximity

    if best_drivable_area:
        x1, y1, x2, y2 = best_drivable_area

        # If bottom of bounding box is near the bottom of the frame, force a stop
        if y2 > 0.95 * h:
            points = np.array([[x1, y2], [x1, y1], [x2, y1], [x2, y2]], np.int32)
            cv2.fillPoly(drivable_area_mask, [points], 255)
        else:
            # Find lane edges near the center
            center_line = w // 2
            lane_edges_y, lane_edges_x = np.where(edges > 0)

            # Filter edges within a reasonable horizontal range near the center
            horizontal_range = w // 8  # Adjust as needed
            valid_edges_x = lane_edges_x[(lane_edges_x > center_line - horizontal_range) & (lane_edges_x < center_line + horizontal_range)]
            valid_edges_y = lane_edges_y[(lane_edges_x > center_line - horizontal_range) & (lane_edges_x < center_line + horizontal_range)]

            # Find the leftmost and rightmost lane edges
            if len(valid_edges_x) > 0:
                leftmost_edge = np.min(valid_edges_x)
                rightmost_edge = np.max(valid_edges_x)

                # Use these edges as the base of the trapezoid
                x1 = leftmost_edge
                x2 = rightmost_edge

            # Dynamic Trapezoid points based on detected drivable area and lane edges
            bottom_width = x2 - x1
            top_width = bottom_width * 0.6  # Adjust this factor as needed
            top_x1 = int(x1 + (bottom_width - top_width) / 2)
            top_x2 = int(x2 - (bottom_width - top_width) / 2)
            top_y = int(y1 + (y2 - y1) * 0.3) # Make the trapezoid higher up
            trapezoid_points = np.array([[x1, y2], [top_x1, top_y], [top_x2, top_y], [x2, y2]], np.int32)

            cv2.fillPoly(drivable_area_mask, [trapezoid_points], 255)

    # Combine drivable area with lane mask
    final_mask = cv2.bitwise_and(lane_mask, drivable_area_mask)
    return final_mask, best_y2, best_drivable_area #Return best y2 for height consideration and best_drivable_area to see if it is being detected

def get_traffic_light_color(frame, x1, y1, x2, y2):
    """Estimates traffic light color based on maximum intensity in the box."""
    traffic_light_crop = frame[y1:y2, x1:x2]

    if traffic_light_crop.size == 0:
        return "unknown"

    # Split the image into its color channels
    b, g, r = cv2.split(traffic_light_crop)

    # Calculate the mean intensity of each channel
    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)

    # Determine the color with the highest intensity
    if mean_r > mean_g and mean_r > mean_b:
        return "red"
    elif mean_g > mean_r and mean_g > mean_b:
        return "green"
    elif mean_b > mean_r and mean_b > mean_g:
        return "blue"  # Rarely happens for traffic lights, but good to have
    else:
        return "unknown"

# Initialize state variable for hysteresis
is_slowing_down = False

# Initialize buffer for smoothing
drivable_bottom_y_buffer = []
buffer_size = 5

def get_driving_command(frame, detections, drivable_area_mask, drivable_bottom_y, best_drivable_area):
    """Determines driving command based on object proximity, traffic light color, and drivable area height."""
    global is_slowing_down
    global drivable_bottom_y_buffer
    h, w = frame.shape[:2]

    height_slow_threshold = int(0.7 * h)  # Example: 70% of frame height
    height_stop_threshold = int(0.85 * h) # Example: 85% of frame height
    hysteresis = 0.05 * h  # Example: 5% of frame height

    # Smoothing:
    drivable_bottom_y_buffer.append(drivable_bottom_y)
    if len(drivable_bottom_y_buffer) > buffer_size:
        drivable_bottom_y_buffer.pop(0)  # Remove the oldest value

    smoothed_drivable_bottom_y = sum(drivable_bottom_y_buffer) / len(drivable_bottom_y_buffer) if drivable_bottom_y_buffer else 0

    #Check if any drivable area was detected:
    if best_drivable_area is None:
        return "STOP!!!! No Drivable Area"

    # Check trapezium height (proximity based on y2 coordinate):
    if smoothed_drivable_bottom_y > height_stop_threshold:
        is_slowing_down = False  # Reset slow down
        return "SLOW!!!! (Drivable Area Very Close)"
    elif smoothed_drivable_bottom_y > height_slow_threshold and not is_slowing_down:
        is_slowing_down = True  # Enter slow down state
        return "SLOW DOWN.... (Drivable Area Close)"
    elif smoothed_drivable_bottom_y <= height_slow_threshold - hysteresis and is_slowing_down:
        is_slowing_down = False # Exit slow down state
        return "GO STRAIGHT+++"
    elif is_slowing_down:
        return "SLOW DOWN.... (Drivable Area Close)"
    else:
        return "GO STRAIGHT+++"

def run_yolo_detection(video_path, model_path, conf_threshold=0.5, start_time=0, end_time=None):
    model = YOLO(model_path, verbose=False)  # Disable YOLO's printouts
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Error: Unable to open video file.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps) if end_time else total_frames

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    text_color = (0, 0, 255)  # Red color in BGR format

    # Initialize counters for detected objects
    car_count = 0
    truck_count = 0
    bus_count = 0
    motorcycle_count = 0
    bicycle_count = 0
    person_count = 0
    
    while cap.isOpened():
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame > end_frame:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame = auto_rotate(frame)
        edges = detect_lane(frame)
        lane_mask = get_lane_mask(edges, frame.shape)

        results = model(frame, conf=conf_threshold, verbose=False) # Disable printing detections

        drivable_area_mask, drivable_bottom_y, best_drivable_area = get_drivable_area_mask(frame, results[0], lane_mask, edges)

        # Get driving command :
        main_command = get_driving_command(frame, results[0], drivable_area_mask, drivable_bottom_y, best_drivable_area)

        # Reset counts for each frame
        car_count = 0
        truck_count = 0
        bus_count = 0
        motorcycle_count = 0
        bicycle_count = 0
        person_count = 0

        # Display boxes around objects (excluding lane and drivable area)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0].item())

            class_names = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "rider", "traffic light", "traffic sign", "lane", "drivable area"]

            if cls < len(class_names) and class_names[cls] not in ["lane", "drivable area"]:
                class_name = class_names[cls]
                color = class_colors.get(cls, (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) # Draw bounding box
                cv2.putText(frame, f"{class_name} {box.conf[0]:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
                
                # Increment the respective counter
                if class_name == "car":
                    car_count += 1
                elif class_name == "truck":
                    truck_count += 1
                elif class_name == "bus":
                    bus_count += 1
                elif class_name == "motorcycle":
                    motorcycle_count += 1
                elif class_name == "bicycle":
                    bicycle_count += 1
                elif class_name == "person":
                    person_count += 1

        # Display the driving command in the bottom left corner
        cv2.putText(frame, main_command, (30, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

        # Display object counts in the top right corner
        text_x = frame.shape[1] - 200
        text_y = 30
        line_height = 20
        white_color = (255, 255, 255)

        cv2.putText(frame, f"Cars: {car_count}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2)
        cv2.putText(frame, f"Trucks: {truck_count}", (text_x, text_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2)
        cv2.putText(frame, f"Buses: {bus_count}", (text_x, text_y + 2 * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2)
        cv2.putText(frame, f"Motorcycles: {motorcycle_count}", (text_x, text_y + 3 * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2)
        cv2.putText(frame, f"Bicycles: {bicycle_count}", (text_x, text_y + 4 * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2)
        cv2.putText(frame, f"Pedestrians: {person_count}", (text_x, text_y + 5 * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2)

        # Visualization:
        frame[edges > 0] = [0, 255, 255]  # Yellow lane markings

        # Create a green overlay for drivable area with 10% opacity
        drivable_area_overlay = np.zeros_like(frame, dtype=np.uint8)
        drivable_area_overlay[drivable_area_mask > 0] = [0, 255, 0] # Green

        # Blend the overlay with the original frame
        frame = cv2.addWeighted(frame, 1, drivable_area_overlay, 0.1, 0) # Opacity is 0.1
        cv2.imshow("YOLOv8 Detection", frame)

        # Check if the window is closed by the user
        if cv2.getWindowProperty("YOLOv8 Detection", cv2.WND_PROP_VISIBLE) < 1:
            break  # Exit the loop

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Execution
input_video = r"C:\Users\garg1\OneDrive\Desktop\PS1\3\traffic_video.avi"  # Change to your .mov or .mp4 file
model_path = r"C:\Users\garg1\OneDrive\Desktop\KrackHacK-akatsuki\Website-rendering\best.pt"
run_yolo_detection(input_video, model_path, conf_threshold=0.5, start_time=10, end_time=30)