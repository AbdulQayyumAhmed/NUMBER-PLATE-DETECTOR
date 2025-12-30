# model.py
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Load YOLO model
model = YOLO("best.pt")

def detect_number_plate(image):
    """
    Input: RGB image as numpy array
    Output: PIL Image with bounding boxes + confidence score
    """

    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)

    # Load font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # YOLO prediction
    results = model.predict(image, conf=0.5, verbose=False)

    for r in results:
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for box, conf in zip(xyxy, confs):
            x1, y1, x2, y2 = map(int, box.tolist())
            label = f"Plate {conf*100:.1f}%"

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline="green", width=3)

            # Get text size using textbbox
            bbox = draw.textbbox((0,0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Draw label background
            draw.rectangle([x1, y1 - text_height, x1 + text_width, y1], fill="green")

            # Draw text
            draw.text((x1, y1 - text_height), label, fill="white", font=font)

    return img_pil
