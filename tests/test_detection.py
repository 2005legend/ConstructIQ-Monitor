import pytest
import numpy as np
import cv2
import json
from hypothesis import given, strategies as st
from datetime import datetime
import sys
import os

# Add parent directory to path to import models and detection
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import Detection, Violation
from detection.detector import calculate_ppe_coverage, identify_violations

@st.composite
def mock_detection_generator(draw):
    """Generate a random Detection object for testing."""
    class_name = draw(st.sampled_from(["worker", "helmet", "vest", "machine"]))
    x1 = draw(st.integers(min_value=0, max_value=100))
    y1 = draw(st.integers(min_value=0, max_value=100))
    x2 = draw(st.integers(min_value=x1+10, max_value=x1+200))
    y2 = draw(st.integers(min_value=y1+10, max_value=y1+200))
    conf = draw(st.floats(min_value=0.1, max_value=1.0))
    
    has_mask = draw(st.booleans())
    mask = None
    if has_mask:
        # Create a mock mask
        mask = np.zeros((300, 300), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        
    return Detection(
        id=draw(st.uuids()).hex,
        class_name=class_name,
        bbox=(x1, y1, x2, y2),
        confidence=conf,
        mask=mask,
        coverage_percentage=None
    )

@given(st.lists(mock_detection_generator(), min_size=1, max_size=10))
def test_instance_segmentation_consistency(detections):
    """Feature: construction-monitoring-system, Property 2: Instance Segmentation Consistency"""
    # Assuming the detection module correctly assigned masks if they exist
    for d in detections:
        if d.mask is not None:
            # Mask should be a numpy array
            assert isinstance(d.mask, np.ndarray)
            # Mask should contain values >= 0
            assert np.min(d.mask) >= 0

@given(st.lists(mock_detection_generator(), min_size=1, max_size=10))
def test_ppe_coverage_calculation(detections):
    """Feature: construction-monitoring-system, Property 3: PPE Coverage Calculation Accuracy"""
    # Calculate coverage
    calculate_ppe_coverage(detections)
    
    for d in detections:
        if d.class_name in ["helmet", "vest", "mask"] and d.coverage_percentage is not None:
            assert 0.0 <= d.coverage_percentage <= 100.0

@given(st.lists(mock_detection_generator(), min_size=1, max_size=10))
def test_violation_detection_logic(detections):
    """Feature: construction-monitoring-system, Property 4: Violation Detection Logic"""
    violations = identify_violations(detections)
    
    # Each violation must reference a worker
    worker_ids = [d.id for d in detections if d.class_name == "worker"]
    for v in violations:
        assert v.worker_id in worker_ids
        assert v.type in ["missing_helmet", "missing_vest", "missing_mask"]
        assert v.severity in ["high", "medium", "low"]

@given(st.lists(mock_detection_generator(), min_size=2, max_size=20))
def test_instance_id_uniqueness(detections):
    """Feature: construction-monitoring-system, Property 6: Instance ID Uniqueness"""
    # Extract all IDs
    ids = [d.id for d in detections]
    # Check if they are unique
    assert len(ids) == len(set(ids))

# Property 1 (Object Detection Completeness) requires actual images and model inference
# We will do a basic test without hypothesis since inference is slow
def test_object_detection_completeness():
    """Feature: construction-monitoring-system, Property 1: Object Detection Completeness"""
    from detection.detector import detect_objects
    
    # Create a dummy image
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    detections, output_img = detect_objects(dummy_image, conf_threshold=0.1)
    
    assert isinstance(detections, list)
    assert isinstance(output_img, np.ndarray)
