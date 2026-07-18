"""Integration tests for end-to-end workflows across all modules."""
import pytest
import numpy as np
import sys
import os
import json
import tempfile
from unittest.mock import patch, MagicMock
from datetime import datetime

# Mock pycolmap before any project import that transitively imports it
sys.modules["pycolmap"] = MagicMock()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import Detection, Violation, CameraPose, ZoneAnalysis
from progress.tracker import analyze_progress
from fusion.projector import project_to_3d, overlay_violations_3d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_detection(class_name: str, bbox, confidence: float = 0.9):
    import uuid
    return Detection(
        id=str(uuid.uuid4()),
        class_name=class_name,
        bbox=bbox,
        confidence=confidence,
        mask=None,
        coverage_percentage=None,
    )


def _make_violation(worker_id: str, location_3d=None):
    import uuid
    return Violation(
        id=str(uuid.uuid4()),
        type="missing_helmet",
        worker_id=worker_id,
        severity="high",
        location_2d=(50, 50),
        location_3d=location_3d,
        confidence=0.85,
        timestamp=datetime.now(),
    )


# ---------------------------------------------------------------------------
# Test 1 — Detection → Violation pipeline
# ---------------------------------------------------------------------------

def test_detection_to_violation_pipeline():
    """Violations produced by identify_violations reference valid worker IDs."""
    from detection.detector import identify_violations

    worker = _make_detection("worker", (10, 10, 100, 100))
    helmet = _make_detection("helmet", (20, 20, 60, 50))
    detections = [worker, helmet]

    violations = identify_violations(detections)

    assert isinstance(violations, list), "identify_violations must return a list"

    worker_ids = {d.id for d in detections if d.class_name == "worker"}
    for v in violations:
        assert v.worker_id in worker_ids, (
            f"Violation references unknown worker_id: {v.worker_id}"
        )


# ---------------------------------------------------------------------------
# Test 2 — Progress end-to-end
# ---------------------------------------------------------------------------

def test_progress_end_to_end():
    """analyze_progress returns a valid ProgressResult for two simple images."""
    img1 = np.full((200, 200, 3), 80, dtype=np.uint8)
    img2 = np.full((200, 200, 3), 80, dtype=np.uint8)
    # Add a distinct white square to img2
    img2[75:125, 75:125] = 255

    result = analyze_progress(img1, img2)

    assert 0.0 <= result.overall_change <= 100.0, (
        f"overall_change {result.overall_change} is outside [0, 100]"
    )
    assert len(result.zone_analysis) == 64, (
        f"Expected 64 zones (8×8 grid), got {len(result.zone_analysis)}"
    )
    assert result.change_overlay.shape == img1.shape, (
        "change_overlay shape must match input image shape"
    )


# ---------------------------------------------------------------------------
# Test 3 — Fusion pipeline with mock pose
# ---------------------------------------------------------------------------

def test_fusion_with_mock_pose_pipeline():
    """project_to_3d + overlay_violations_3d round-trip works correctly."""
    pose = CameraPose(
        image_id="mock_cam.jpg",
        position=(0.0, 0.0, 0.0),   # zero translation: camera at world origin
        rotation=(1.0, 0.0, 0.0, 0.0),  # identity quaternion (COLMAP wxyz)
        confidence=1.0,
        intrinsics=np.array([800.0, 800.0, 400.0, 300.0]),
    )

    point_3d = project_to_3d((400, 300), pose, depth=5.0)

    # Must be a 3-tuple of floats
    assert len(point_3d) == 3, "project_to_3d must return a 3-tuple"
    assert all(isinstance(v, float) for v in point_3d), (
        "All elements of the 3D point must be floats"
    )

    # Build a violation with the projected location
    violation = _make_violation(worker_id="worker_001", location_3d=point_3d)

    markers = overlay_violations_3d(None, [violation])

    # The violation has a 3D location so it must appear in 3d_markers
    assert len(markers["3d_markers"]) == 1, (
        "Violation with location_3d must appear in '3d_markers'"
    )
    assert markers["3d_markers"][0]["position"] == point_3d


# ---------------------------------------------------------------------------
# Test 4 — Data consistency across modules (JSON serialisability)
# ---------------------------------------------------------------------------

class _DatetimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects and numpy arrays."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


def test_data_consistency_across_modules():
    """ProgressResult-like and DetectionReport-like dicts can be serialised."""
    progress_dict = {
        "overall_change": 12.5,
        "timestamp": datetime.now(),
        "zones": [
            {"zone_id": "A1", "coordinates": (0, 0, 50, 50), "change_percentage": 10.0, "change_type": "minor_change"}
        ],
        "before_image": np.zeros((10, 10, 3), dtype=np.uint8),
        "after_image": np.zeros((10, 10, 3), dtype=np.uint8),
        "change_overlay": np.zeros((10, 10, 3), dtype=np.uint8),
    }

    detection_dict = {
        "session_id": "test-session-001",
        "timestamp": datetime.now(),
        "total_detections": 2,
        "violation_count": 1,
        "detections": [],
        "violations": [],
        "processing_time": 0.123,
    }

    progress_json = json.dumps(progress_dict, cls=_DatetimeEncoder)
    detection_json = json.dumps(detection_dict, cls=_DatetimeEncoder)

    assert progress_json, "ProgressResult JSON must not be empty"
    assert detection_json, "DetectionReport JSON must not be empty"

    # Confirm they round-trip back to dicts
    assert json.loads(progress_json)["overall_change"] == 12.5
    assert json.loads(detection_json)["session_id"] == "test-session-001"
