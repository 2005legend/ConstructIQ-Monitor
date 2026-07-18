import os
import uuid
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from typing import List, Tuple, Dict, Any
from models import Detection, Violation

# We will use the models that exist or fallback to a standard YOLO segmentation model
MODEL_PATH = "best.pt"
FALLBACK_MODEL = "yolo11n-seg.pt"

if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    model = YOLO(FALLBACK_MODEL)

def preprocess(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE for contrast enhancement."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

def detect_objects(image: np.ndarray, conf_threshold: float = 0.25) -> Tuple[List[Detection], np.ndarray]:
    """
    Performs object detection and instance segmentation using YOLO11.
    Returns the detections and the plotted image.
    """
    processed_img = preprocess(image)
    results = model(processed_img, conf=conf_threshold)
    
    detections = []
    output_image = results[0].plot()
    
    names = model.names
    
    if results[0].boxes:
        for idx, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Extract mask if available
            mask = None
            if results[0].masks is not None and len(results[0].masks) > idx:
                # Mask data
                mask_data = results[0].masks.data[idx].cpu().numpy()
                # Resize to match original image dimensions
                mask = cv2.resize(mask_data, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            det = Detection(
                id=str(uuid.uuid4()),
                class_name=cls_name,
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                mask=mask,
                coverage_percentage=None
            )
            detections.append(det)
            
    return detections, output_image

def calculate_ppe_coverage(detections: List[Detection]) -> None:
    """
    Computes body coverage percentages for PPE compliance.
    Updates the coverage_percentage field in-place for PPE detections.
    """
    # Assuming PPE items are 'helmet', 'vest', 'mask' etc.
    # To calculate coverage, we would need to associate PPE masks with worker masks
    # For now, we compute the ratio of PPE mask area to worker mask area if associated,
    # or just set a placeholder. If there's no mask, we calculate based on bbox area.
    
    workers = [d for d in detections if d.class_name.lower() in ["worker", "person"]]
    ppe_items = [d for d in detections if d.class_name.lower() in ["helmet", "hardhat", "vest", "safety vest", "mask"]]
    
    for ppe in ppe_items:
        # Find closest worker
        closest_worker = None
        min_dist = float('inf')
        ppe_center = ((ppe.bbox[0] + ppe.bbox[2]) / 2, (ppe.bbox[1] + ppe.bbox[3]) / 2)
        
        for w in workers:
            w_center = ((w.bbox[0] + w.bbox[2]) / 2, (w.bbox[1] + w.bbox[3]) / 2)
            dist = (ppe_center[0] - w_center[0])**2 + (ppe_center[1] - w_center[1])**2
            if dist < min_dist:
                # Basic check if PPE center is within worker bbox
                if (w.bbox[0] <= ppe_center[0] <= w.bbox[2]) and (w.bbox[1] <= ppe_center[1] <= w.bbox[3]):
                    min_dist = dist
                    closest_worker = w
                    
        if closest_worker:
            if ppe.mask is not None and closest_worker.mask is not None:
                ppe_area = np.sum(ppe.mask > 0)
                worker_area = np.sum(closest_worker.mask > 0)
                if worker_area > 0:
                    coverage = (ppe_area / worker_area) * 100
                    ppe.coverage_percentage = min(100.0, max(0.0, coverage))
                else:
                    ppe.coverage_percentage = 0.0
            else:
                # Bbox based
                ppe_area = (ppe.bbox[2] - ppe.bbox[0]) * (ppe.bbox[3] - ppe.bbox[1])
                worker_area = (closest_worker.bbox[2] - closest_worker.bbox[0]) * (closest_worker.bbox[3] - closest_worker.bbox[1])
                if worker_area > 0:
                    coverage = (ppe_area / worker_area) * 100
                    ppe.coverage_percentage = min(100.0, max(0.0, coverage))
                else:
                    ppe.coverage_percentage = 0.0

def identify_violations(detections: List[Detection]) -> List[Violation]:
    """
    Flags missing helmets, vests, and masks.
    """
    violations = []
    
    workers = [d for d in detections if d.class_name.lower() in ["worker", "person"]]
    # Map class names to standardized PPE names
    ppe_items = []
    for d in detections:
        name = d.class_name.lower()
        if "helmet" in name or "hardhat" in name:
            ppe_items.append((d, "helmet"))
        elif "vest" in name:
            ppe_items.append((d, "vest"))
        elif "mask" in name:
            ppe_items.append((d, "mask"))
            
    for worker in workers:
        w_center = ((worker.bbox[0] + worker.bbox[2]) / 2, (worker.bbox[1] + worker.bbox[3]) / 2)
        
        has_helmet = False
        has_vest = False
        
        for ppe, ppe_type in ppe_items:
            # Check overlap or containment
            ppe_center = ((ppe.bbox[0] + ppe.bbox[2]) / 2, (ppe.bbox[1] + ppe.bbox[3]) / 2)
            # Simple check: PPE center is within worker bbox
            if (worker.bbox[0] <= ppe_center[0] <= worker.bbox[2]) and (worker.bbox[1] <= ppe_center[1] <= worker.bbox[3]):
                if ppe_type == "helmet":
                    has_helmet = True
                elif ppe_type == "vest":
                    has_vest = True
                    
        # Check requirements
        if not has_helmet:
            violations.append(Violation(
                id=str(uuid.uuid4()),
                type="missing_helmet",
                worker_id=worker.id,
                severity="high",
                location_2d=(int(w_center[0]), int(w_center[1])),
                location_3d=None,
                confidence=worker.confidence,
                timestamp=datetime.now()
            ))
            
        if not has_vest:
            violations.append(Violation(
                id=str(uuid.uuid4()),
                type="missing_vest",
                worker_id=worker.id,
                severity="medium",
                location_2d=(int(w_center[0]), int(w_center[1])),
                location_3d=None,
                confidence=worker.confidence,
                timestamp=datetime.now()
            ))
            
    return violations
