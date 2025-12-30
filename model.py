# model.py
from ultralytics import YOLO
import cv2
import numpy as np

# Load your trained YOLO model
model = YOLO("best.pt")

def detect_number_plate(image):
    """
    Input: image as numpy array (RGB)
    Output: BGR image with bounding boxes + confidence score
    """

    results = model.predict(image, conf=0.5, verbose=False)

    # Convert copy to BGR for drawing
    img_out = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for r in results:
        boxes = r.boxes

        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()   # confidence values

        for box, conf in zip(xyxy, confs):
            x1, y1, x2, y2 = map(int, box.tolist())

            # Draw bounding box
            cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # Prepare confidence label
            conf_percent = f"{conf*100:.1f}%"  # convert 0.92 → "92.0%"

            label = f"Plate {conf_percent}"

            # Draw label above the box
            cv2.putText(img_out, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return img_out
