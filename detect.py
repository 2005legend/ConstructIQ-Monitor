from ultralytics import YOLO
import cv2
import os
import numpy as np

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
        
    return output_image, counts
