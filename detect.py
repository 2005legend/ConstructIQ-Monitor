from ultralytics import YOLO
import cv2
import os
import numpy as np

HAZARD_RULES = {
    "NO-Hardhat": "⚠️ Worker without helmet detected",
    "NO-Safety Vest": "⚠️ Worker without vest detected",
    "NO-Mask": "⚠️ Worker without mask detected",
}

def preprocess(image):
    # CLAHE for contrast enhancement (same as Vision-to-Voice)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

def check_hazards(counts):
    alerts = []
    for cls, rule in HAZARD_RULES.items():
        if counts.get(cls, 0) > 0:
            alerts.append(f"{rule} — Count: {counts[cls]}")
    return alerts

# Use the trained model if available, else fallback to a pretrained model for demo
MODEL_PATH = "best.pt"
FALLBACK_MODEL = "yolo11n.pt"

if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    model = YOLO(FALLBACK_MODEL)

def run_inference(image):
    """
    Runs YOLO11 inference on a given image.
    Args:
        image: PIL Image or OpenCV image array.
    Returns:
        output_image: Image with bounding boxes drawn.
        counts: Dictionary of detected class counts.
    """
    # Preprocess image
    image = preprocess(image)
    
    results = model(image)
    
    # Get the plotted image (numpy array)
    output_image = results[0].plot()
    
    # Get counts
    counts = {}
    names = model.names
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = names[cls_id]
        counts[cls_name] = counts.get(cls_name, 0) + 1
        
    hazards = check_hazards(counts)
        
    return output_image, counts, hazards
