import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
import tempfile
import os
import time
import torch
from vidgear.gears import CamGear

# Shared Class Colors (consistent across both modes)
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

# --- DRIVING ASSISTANCE FUNCTIONS ---
def auto_rotate(frame):
    h, w = frame.shape[:2]
    return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) if h > w else frame

def detect_lane(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

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

def run_yolo_detection(frame, model, conf_threshold=0.25):
    try:
        edges = detect_lane(frame)
        lane_mask = get_lane_mask(edges, frame.shape)

        results = model(frame, conf=conf_threshold, verbose=False) # Disable printing detections

        drivable_area_mask, drivable_bottom_y, best_drivable_area = get_drivable_area_mask(frame, results[0], lane_mask, edges)

        # Get driving command :
        main_command = get_driving_command(frame, results[0], drivable_area_mask, drivable_bottom_y, best_drivable_area)

        # Initialize counters for detected objects
        car_count = 0
        truck_count = 0
        bus_count = 0
        motorcycle_count = 0
        bicycle_count = 0
        person_count = 0

        annotated_frame = frame.copy()

        # Display boxes around objects (excluding lane and drivable area)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0].item())

            class_names = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "rider", "traffic light", "traffic sign", "lane", "drivable area"]

            if cls < len(class_names) and class_names[cls] not in ["lane", "drivable area"]:
                class_name = class_names[cls]
                color = class_colors.get(cls, (255, 255, 255))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2) # Draw bounding box
                cv2.putText(annotated_frame, f"{class_name} {box.conf[0]:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
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
        cv2.putText(annotated_frame, main_command, (30, annotated_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Display object counts in the top right corner
        text_x = annotated_frame.shape[1] - 200
        text_y = 30
        line_height = 20
        white_color = (255, 255, 255)

        cv2.putText(annotated_frame, f"Cars: {car_count}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2)
        cv2.putText(annotated_frame, f"Trucks: {truck_count}", (text_x, text_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2)
        cv2.putText(annotated_frame, f"Buses: {bus_count}", (text_x, text_y + 2 * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2)
        cv2.putText(annotated_frame, f"Motorcycles: {motorcycle_count}", (text_x, text_y + 3 * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2)
        cv2.putText(annotated_frame, f"Bicycles: {bicycle_count}", (text_x, text_y + 4 * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2)
        cv2.putText(annotated_frame, f"Pedestrians: {person_count}", (text_x, text_y + 5 * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white_color, 2)

        # Visualization:
        annotated_frame[edges > 0] = [0, 255, 255]  # Yellow lane markings

        # Create a green overlay for drivable area with 10% opacity
        drivable_area_overlay = np.zeros_like(annotated_frame, dtype=np.uint8)
        drivable_area_overlay[drivable_area_mask > 0] = [0, 255, 0] # Green

        # Blend the overlay with the original frame
        annotated_frame = cv2.addWeighted(annotated_frame, 1, drivable_area_overlay, 0.1, 0) # Opacity is 0.1

        return annotated_frame, car_count, truck_count, bus_count, motorcycle_count, bicycle_count, person_count, main_command, edges, drivable_area_mask

    except Exception as e:
        st.error(f"Error in run_yolo_detection: {e}")
        return frame, 0, 0, 0, 0, 0, 0, "ERROR", detect_lane(frame), np.zeros_like(frame[:, :, 0], dtype=np.uint8)  # Return defaults

def process_video1(source, model, options=None):
    if source == 'Use camera':
        cap = cv2.VideoCapture(0)
        is_camgear = False
    elif source.startswith('http'):  # YouTube source
        cap = CamGear(source=source, stream_mode=True, logging=True, **options).start()
        is_camgear = True
    else:  # Local file
        cap = cv2.VideoCapture(source)
        is_camgear = False

    FRAME_WINDOW = st.empty()
    counters_placeholder = st.empty()  # Placeholder for counters
    command_placeholder = st.empty()  # Placeholder for GO STOP SLOW commands

    try:
        while True:
            if is_camgear:
                try:
                    frame = cap.read()
                except Exception as e:
                    st.error(f"CamGear read error: {e}")
                    break  # Exit loop on read error
                if frame is None:
                    break
            else:
                try:
                    ret, frame = cap.read()
                except Exception as e:
                    st.error(f"VideoCapture read error: {e}")
                    break  # Exit loop on read error
                if not ret:
                    break

            if frame is None or frame.size == 0:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = auto_rotate(frame)
            annotated_frame, car_count, truck_count, bus_count, motorcycle_count, bicycle_count, person_count, main_command, edges, drivable_area_mask = run_yolo_detection(frame, model, conf_threshold=0.25)

            # Display driving instructions first
            command_placeholder.markdown(f"""
                <h2 style="text-align: center; color: red;">🚦 Driving Command: <b>{main_command}</b> 🚦</h2>
            """, unsafe_allow_html=True)

            # Display object counts in Streamlit
            counters_placeholder.markdown(f"""
                Cars: {car_count}  
                Trucks: {truck_count}  
                Buses: {bus_count}  
                Motorcycles: {motorcycle_count}  
                Bicycles: {bicycle_count}  
                Pedestrians: {person_count}
            """)

            # Show the annotated video frame
            FRAME_WINDOW.image(annotated_frame, channels="RGB")

    except Exception as e:
        st.error(f"An error occurred in process_video1: {e}")

    finally:
        if is_camgear:
            try:
                cap.stop()  # Use stop() for CamGear
            except Exception as e:
                st.error(f"Error stopping CamGear: {e}")
        else:
            if isinstance(cap, cv2.VideoCapture) and cap.isOpened():
                try:
                    cap.release()
                except Exception as e:
                    st.error(f"Error releasing VideoCapture: {e}")

        if isinstance(source, str) and os.path.exists(source):
            try:
                os.remove(source)  # Cleanup after processing
            except Exception as e:
                st.error(f"Error removing temp file: {e}")

# --- ROAD MONITOR FUNCTIONS ---

# Global Variables (MUST be initialized outside the function)
tracked_vehicles = {}
TRAFFIC_SMOOTHING_WINDOW = 5
traffic_history = []
vehicle_arrival_times = []
last_traffic_update = time.time()
traffic_level = "Low"
class_names = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]
vehicle_classes = {0, 1, 2, 3, 4}
class_counts = {name: 0 for name in class_names}
counted_vehicle_ids = {}  # Store last seen time for each vehicle ID
RECOUNT_DELAY = 5  # Time in seconds before a vehicle ID can be recounted

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

def run_static_camera_detection(frame, model, conf_threshold=0.3, iou_threshold=0.5):
    global tracked_vehicles, last_traffic_update, traffic_level, traffic_history, class_counts, vehicle_arrival_times, counted_vehicle_ids

    frame_height, frame_width = frame.shape[:2]

    roi_x1 = 0
    roi_y1 = int(frame_height * 0.5)  # Bottom half ROI start
    roi_x2 = frame_width
    roi_y2 = frame_height


    results = model.track(frame, persist=True, conf=conf_threshold, iou=iou_threshold, imgsz=1280, verbose=False)

    annotated_frame = frame.copy()

    if results and results[0] and results[0].boxes:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            obj_id = int(box.id[0].item()) #Get Object ID

            box_area = (x2 - x1) * (y2 - y1)

            if cls >= len(class_names) or conf < conf_threshold:
                continue

            if cls in vehicle_classes:
                # Limit bounding box display and counting to bottom half (and check ROI)
                if y1 > frame_height * 0.5 and roi_x1 < x1 and roi_x2 > x2 and roi_y1 < y1 and roi_y2 > y2:
                    color = class_colors.get(cls, (255, 255, 255))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2) # draw bound box
                    cv2.putText(annotated_frame, f"{class_names[cls]} {conf:.2f} ID: {obj_id}", (x1, y1 - 10), #draw text
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

                    # Count each vehicle ID only once, or after a time delay
                    current_time = time.time()
                    if obj_id not in counted_vehicle_ids or (current_time - counted_vehicle_ids[obj_id] > RECOUNT_DELAY and box_area > 5000) : #Check box are also
                        counted_vehicle_ids[obj_id] = current_time
                        class_counts[class_names[cls]] += 1


        current_time = time.time()
        time_window = 1

        #This counting will not work as i wanted it to cause each model track has their own set of id, and the model keeps learning so it changes the id if the object is at different place, we need to solve the yolo tracking id problem first, one thing that can be done is to add the id to every car or object, but lets keep these lines so you can add it later,
        arrival_times_in_window = [t for t in vehicle_arrival_times if current_time - t <= time_window]
        cars_per_second = len(arrival_times_in_window) / time_window

        traffic_level = classify_traffic(cars_per_second)
        traffic_history.append(traffic_level)

        if len(traffic_history) > TRAFFIC_SMOOTHING_WINDOW:
            traffic_history.pop(0)

        smoothed_traffic = max(set(traffic_history), key=traffic_history.count)
        traffic_level = smoothed_traffic

    # Display traffic and pedestrian crossing info in the top-left
    cv2.putText(annotated_frame, f"Traffic: {traffic_level}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(annotated_frame, f"Pedestrian: {classify_pedestrian_difficulty(traffic_level)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # Display class counts in the top-right
    y_offset = 40
    for i, name in enumerate(class_names):
        text = f"{name}: {class_counts[name]}"
        cv2.putText(annotated_frame, text, (frame_width - 200, y_offset + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2) #Remove ROI display box
    return annotated_frame, traffic_level, classify_pedestrian_difficulty(traffic_level)

def process_video2(source, model, options=None):
    global tracked_vehicles, TRAFFIC_SMOOTHING_WINDOW, traffic_history, vehicle_arrival_times, last_traffic_update, traffic_level, class_names, vehicle_classes, class_counts, counted_vehicle_ids

    FRAME_WINDOW = st.empty()
    INFO_WINDOW = st.empty()
    cap = None
    is_camgear = False

    # Reset global variables at the start of each processing
    tracked_vehicles = {}
    traffic_history = []
    vehicle_arrival_times = []
    last_traffic_update = time.time()
    traffic_level = "Low"
    class_counts = {name: 0 for name in class_names}
    counted_vehicle_ids = {} # Reset counted vehicle IDs

    try:
        if source == 'Use camera':
            cap = cv2.VideoCapture(0)
            is_camgear = False
        elif source.startswith('http'):
            cap = CamGear(source=source, stream_mode=True, logging=True, **(options or {})).start()
            is_camgear = True
        else:
            cap = cv2.VideoCapture(source)
            is_camgear = False

        frame_count = 0
        start_time = time.time()

        while True:
            if is_camgear:
                try:
                    frame = cap.read()
                except Exception as e:
                    st.error(f"CamGear read error: {e}")
                    break

                if frame is None:
                    st.warning("Failed to read frame from CamGear. Retrying...")
                    continue
            else:
                try:
                    ret, frame = cap.read()
                except Exception as e:
                    st.error(f"VideoCapture read error: {e}")
                    break
                if not ret:
                    st.warning("End of video")
                    break

            if frame is None or frame.size == 0:
                continue

            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = auto_rotate(frame)  # Apply auto-rotation here
                annotated_frame, traffic_level, pedestrian_difficulty = run_static_camera_detection(frame, model, conf_threshold = 0.3)
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    fps = frame_count / elapsed_time
                else:
                    fps = 0

                # Display the processed frame
                FRAME_WINDOW.image(annotated_frame, channels="RGB")

                INFO_WINDOW.markdown(f"""
                        Traffic Concentration: {traffic_level}
                        Pedestrian Crossing Difficulty: {pedestrian_difficulty}
                        FPS: {fps:.2f}
                    """)
            except Exception as e:
                st.error(f"Error processing frame in process_video2: {e}")

    except Exception as e:
        st.error(f"An error occurred in process_video2: {e}")

    finally:
        # Release resources
        if is_camgear:
            try:
                if cap is not None:
                    cap.stop()
            except Exception as e:
                st.error(f"Error stopping CamGear: {e}")
        else:
            if isinstance(cap, cv2.VideoCapture) and cap.isOpened():
                try:
                    cap.release()
                except Exception as e:
                    st.error(f"Error releasing VideoCapture: {e}")

        if isinstance(source, str) and os.path.exists(source):
            try:
                os.remove(source)
            except Exception as e:
                st.error(f"Error removing temp file: {e}")

# --- Streamlit App ---

# Load the YOLOv8 model
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO('best.pt')  # Load model initially on CPU
    model.to(device)  # Move model to selected device

    print(f"Model loaded on {device}")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None  # Set model to None to prevent further errors

# Streamlit UI setup
st.set_page_config(page_title="Akatsuki Coders", page_icon="🎥", layout="wide", initial_sidebar_state="expanded")

st.title('Traffic Intelligence Perception')

# Sidebar
with st.sidebar:
    model_source = st.radio('Select Model', ['Driving Assistance', 'Road Monitor'])
    video_source = st.radio('Select video source', ['Use camera', 'YouTube', 'Upload from local'])
    youtube_link = ""
    uploaded_file = None

    if video_source == 'YouTube':
        youtube_link = st.text_input('YouTube video link', '')
    elif video_source == 'Upload from local':
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

# Set desired quality
options = {"STREAM_RESOLUTION": "720p"}

# Process video
if st.button('Start') and model is not None: # Check if model is loaded
    if video_source == 'YouTube' and youtube_link:
        if model_source == "Driving Assistance":
            process_video1(youtube_link, model, options)
        else:
            process_video2(youtube_link, model, options)

    elif video_source == 'Use camera':
        if model_source == "Driving Assistance":
            process_video1('Use camera', model)
        else:
            process_video2('Use camera', model)

    elif video_source == 'Upload from local' and uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        try:
            if model_source == "Driving Assistance":
                process_video1(temp_file_path, model)
            else:
                process_video2(temp_file_path, model)
        finally:
            pass  # Ensure cleanup
elif model is None:
    st.error("Model loading failed. Please check the error messages above.")

# Note on closing resources
st.sidebar.markdown("### About")
st.sidebar.markdown('''
- Our project utilizes deep learning for advanced traffic object detection.
- We trained a YOLO-based model on 70,023 images and validated it with 9,977 images.
- The model detects traffic lights, traffic signals, cars, trucks, motorcycles, driving areas, lanes, riders, bicycles, and buses.
- It includes two specialized models for different scenarios:
  - Road monitor - Designed for analyzing still images.
  - Driving assistance - Optimized for real-time video processing.
- The system generates output videos with bounding boxes around detected elements.
- This technology enhances traffic monitoring, autonomous navigation, and smart city applications.
''')