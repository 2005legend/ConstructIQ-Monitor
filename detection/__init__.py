"""Detection Module - YOLO11 based object detection and PPE compliance analysis."""
from .detector import detect_objects, calculate_ppe_coverage, identify_violations

__all__ = ["detect_objects", "calculate_ppe_coverage", "identify_violations"]
