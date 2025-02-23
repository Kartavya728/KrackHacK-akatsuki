import json
import os

# Set input/output directories
json_folder = r"C:\Users\garg1\OneDrive\Desktop\100K\dataset\test\ann"  # Folder with JSON annotation files
output_folder = r"C:\Users\garg1\OneDrive\Desktop\100K\dataset\test\labels"  # Folder to save YOLO .txt files

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Class mapping for YOLO labels
class_map = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "motorcycle": 3,
    "bicycle": 4,
    "person": 5,
    "rider": 6,
    "traffic light": 7,
    "traffic sign": 8,
    "lane": 9,
    "drivable area": 10
}

# Function to convert polygon to bounding box
def polygon_to_bbox(points):
    x_min = min(p[0] for p in points)
    y_min = min(p[1] for p in points)
    x_max = max(p[0] for p in points)
    y_max = max(p[1] for p in points)
    return x_min, y_min, x_max, y_max

# Process all JSON files
for json_file in os.listdir(json_folder):
    if json_file.endswith(".json"):
        json_path = os.path.join(json_folder, json_file)

        # Load JSON file
        with open(json_path, "r") as f:
            data = json.load(f)

        # Get image dimensions
        img_width = float(data["size"]["width"])
        img_height = float(data["size"]["height"])

        # YOLO annotations list
        yolo_annotations = []

        # Process each object in JSON
        for obj in data["objects"]:
            class_name = obj["classTitle"].lower()  # Convert class name to lowercase
            if class_name not in class_map:
                continue  # Skip unknown classes

            class_id = class_map[class_name]
            points = obj["points"]["exterior"]

            # Handle rectangles and polygons
            if len(points) == 2:  # Rectangle format
                x_min, y_min = points[0]
                x_max, y_max = points[1]
            else:  # Polygon format (convert to bounding box)
                x_min, y_min, x_max, y_max = polygon_to_bbox(points)

            # Convert to YOLO format (normalized)
            x_center = ((x_min + x_max) / 2.0) / img_width
            y_center = ((y_min + y_max) / 2.0) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height

            # Ensure values are within [0,1] bounds
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))

            # Append YOLO annotation
            yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # Save to YOLO .txt file
        txt_filename = os.path.join(output_folder, json_file.replace(".json", ".txt"))
        with open(txt_filename, "w") as f:
            f.write("\n".join(yolo_annotations))

        print(f"✅ Converted {json_file} -> {txt_filename}")

print("\n🎯 All JSON files converted to YOLO format successfully!")