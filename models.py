from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
from datetime import datetime
import numpy as np

# --- Detection Models ---
@dataclass
class Detection:
    id: str
    class_name: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    mask: Optional[np.ndarray]
    coverage_percentage: Optional[float]

@dataclass
class Violation:
    id: str
    type: str  # "missing_helmet", "missing_vest", "missing_mask"
    worker_id: str
    severity: str  # "high", "medium", "low"
    location_2d: Tuple[int, int]
    location_3d: Optional[Tuple[float, float, float]]
    confidence: float
    timestamp: datetime
    image_id: Optional[str] = None

# --- Progress Models ---
@dataclass
class ZoneAnalysis:
    zone_id: str
    coordinates: Tuple[int, int, int, int]
    change_percentage: float
    change_type: str  # "construction", "removal", "no_change"

@dataclass
class ProgressResult:
    overall_change: float
    zone_analysis: List[ZoneAnalysis]
    before_image: np.ndarray
    after_image: np.ndarray
    change_overlay: np.ndarray
    timestamp: datetime

# --- SfM Models ---
@dataclass
class CameraPose:
    image_id: str
    position: Tuple[float, float, float]
    rotation: Tuple[float, float, float, float]
    confidence: float
    intrinsics: Optional[np.ndarray]

@dataclass
class ReconstructionResult:
    success: bool
    camera_poses: List[CameraPose]
    point_cloud_path: str
    quality_score: float
    error_message: Optional[str]
    processing_time: float

# --- Export Models ---
@dataclass
class DetectionReport:
    session_id: str
    timestamp: datetime
    total_detections: int
    violation_count: int
    detections: List[Detection]
    violations: List[Violation]
    processing_time: float

@dataclass
class ProgressReport:
    session_id: str
    timestamp: datetime
    overall_change: float
    zones: List[ZoneAnalysis]
    comparison_metadata: dict
    processing_time: float

# --- Core System Models ---
@dataclass
class Session:
    session_id: str
    timestamp: datetime
    image_paths: List[str]
    reconstruction: Optional[ReconstructionResult] = None
    detections: Optional[List[Detection]] = None
    violations: Optional[List[Violation]] = None
    
    @property
    def has_3d(self) -> bool:
        return self.reconstruction is not None and self.reconstruction.success
        
    @property
    def has_detections(self) -> bool:
        return self.detections is not None
