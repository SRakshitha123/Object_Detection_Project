
from ultralytics import YOLO
import cv2
import json
import os
import urllib.request

# Create input and output folders
os.makedirs("images", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Download a sample image
image_path = "images/bus.jpg"

if not os.path.exists(image_path):
    url = "https://ultralytics.com/images/bus.jpg"
    urllib.request.urlretrieve(url, image_path)

# Load a pretrained YOLO model
model = YOLO("yolo11n.pt")

# Detect objects in the image
results = model(image_path, conf=0.25)

# Read the original image
image = cv2.imread(image_path)

if image is None:
    raise ValueError("Could not load the input image")

detections = []

# Process detected objects
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int, box.xyxy[0].tolist()
        )

        # Draw bounding box
        cv2.rectangle(
            image, (x1, y1), (x2, y2),
            (0, 255, 0), 2
        )

        # Draw object label
        cv2.putText(
            image,
            f"{label} {confidence:.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )

        # Store detection details
        detections.append({
            "label": label,
            "confidence": round(confidence, 4),
            "bounding_box": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
        })

# Save annotated image
cv2.imwrite("output/annotated.jpg", image)

# Save detections as JSON
with open("output/detections.json", "w") as file:
    json.dump(detections, file, indent=4)

print("Object detection completed!")
print("Annotated image: output/annotated.jpg")
print("JSON results: output/detections.json")
print("Number of objects detected:", len(detections))
