"""Property-based and unit tests for the camera calibration module."""
import pytest
import numpy as np
from hypothesis import given, settings, strategies as st
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import CameraPose
from sfm.calibrator import (
    estimate_camera_parameters,
    validate_pose_confidence,
    filter_valid_poses,
    diagnose_pose_failure,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_pose(confidence: float, intrinsics=None, position=(0.0, 1.0, 0.0)):
    """Factory for CameraPose objects used across tests."""
    return CameraPose(
        image_id="test.jpg",
        position=position,
        rotation=(1.0, 0.0, 0.0, 0.0),
        confidence=confidence,
        intrinsics=intrinsics,
    )


# ---------------------------------------------------------------------------
# Property 49 — Camera parameter estimation
# ---------------------------------------------------------------------------

@given(st.integers(320, 4096), st.integers(240, 4096))
def test_camera_parameter_estimation_property_49(width, height):
    """For any valid image dimensions, estimate_camera_parameters must return
    an array of length 4 with all values > 0, cx == width/2, cy == height/2.
    """
    params = estimate_camera_parameters(width, height)

    assert len(params) == 4, "Must return exactly 4 values [fx, fy, cx, cy]"
    assert all(v > 0 for v in params), "All intrinsic parameters must be positive"
    assert params[2] == width / 2.0, "cx must equal width / 2"
    assert params[3] == height / 2.0, "cy must equal height / 2"


# ---------------------------------------------------------------------------
# Property 51 — Pose confidence is always in [0, 1]
# ---------------------------------------------------------------------------

@given(st.floats(0.0, 1.0))
@settings(max_examples=200)
def test_pose_confidence_metrics_property_51(initial_confidence):
    """After validate_pose_confidence, the result confidence must be in [0, 1]."""
    intrinsics = np.array([800.0, 800.0, 400.0, 300.0])
    pose = _make_pose(initial_confidence, intrinsics=intrinsics)

    validated = validate_pose_confidence(pose)

    assert 0.0 <= validated.confidence <= 1.0, (
        f"Confidence {validated.confidence} is outside [0, 1]"
    )


# ---------------------------------------------------------------------------
# filter_valid_poses
# ---------------------------------------------------------------------------

def test_filter_valid_poses_property():
    """filter_valid_poses must only return poses at or above the threshold."""
    confidences = [0.1, 0.3, 0.5, 0.7, 0.9, 0.0, 1.0]
    poses = [_make_pose(c) for c in confidences]

    threshold = 0.5
    filtered = filter_valid_poses(poses, min_confidence=threshold)

    assert all(p.confidence >= threshold for p in filtered), (
        "Filtered list contains poses below the threshold"
    )
    # Check that we kept the right ones
    expected_count = sum(1 for c in confidences if c >= threshold)
    assert len(filtered) == expected_count


def test_filter_valid_poses_empty():
    """filter_valid_poses on an empty list returns an empty list."""
    assert filter_valid_poses([]) == []


def test_filter_valid_poses_all_filtered():
    """filter_valid_poses returns empty list when no pose meets threshold."""
    poses = [_make_pose(0.1), _make_pose(0.2)]
    assert filter_valid_poses(poses, min_confidence=0.9) == []


# ---------------------------------------------------------------------------
# diagnose_pose_failure
# ---------------------------------------------------------------------------

def test_diagnose_pose_failure_returns_dict():
    """diagnose_pose_failure must return a dict with non-empty cause and resolution."""
    pose = _make_pose(confidence=0.1)
    result = diagnose_pose_failure(pose)

    assert isinstance(result, dict), "Result must be a dict"
    assert "cause" in result, "Result must have 'cause' key"
    assert "resolution" in result, "Result must have 'resolution' key"
    assert isinstance(result["cause"], str) and result["cause"], "'cause' must be a non-empty string"
    assert isinstance(result["resolution"], str) and result["resolution"], "'resolution' must be a non-empty string"


def test_diagnose_low_confidence():
    """Low confidence pose should report the correct cause."""
    pose = _make_pose(confidence=0.1)
    result = diagnose_pose_failure(pose)
    assert result["cause"] == "Low confidence pose"


def test_diagnose_missing_intrinsics():
    """Missing intrinsics should be detected when confidence is acceptable."""
    pose = _make_pose(confidence=0.6, intrinsics=None)
    result = diagnose_pose_failure(pose)
    assert result["cause"] == "Missing camera intrinsics"


def test_diagnose_zero_position():
    """Zero position with acceptable confidence and intrinsics gives the right diagnosis."""
    intrinsics = np.array([800.0, 800.0, 400.0, 300.0])
    pose = _make_pose(confidence=0.6, intrinsics=intrinsics, position=(0.0, 0.0, 0.0))
    result = diagnose_pose_failure(pose)
    assert result["cause"] == "Zero position — reconstruction may have failed"


def test_diagnose_unknown_for_healthy_pose():
    """A healthy pose returns 'Unknown' cause."""
    intrinsics = np.array([800.0, 800.0, 400.0, 300.0])
    pose = _make_pose(confidence=0.8, intrinsics=intrinsics, position=(1.0, 2.0, 3.0))
    result = diagnose_pose_failure(pose)
    assert result["cause"] == "Unknown"


# ---------------------------------------------------------------------------
# Confidence clamping edge cases
# ---------------------------------------------------------------------------

def test_pose_confidence_clamped():
    """validate_pose_confidence must clamp confidence to [0.0, 1.0] at edges."""
    intrinsics = np.array([800.0, 800.0, 400.0, 300.0])

    for edge in (0.0, 1.0):
        pose = _make_pose(edge, intrinsics=intrinsics)
        validated = validate_pose_confidence(pose)
        assert 0.0 <= validated.confidence <= 1.0, (
            f"Edge case confidence={edge} produced {validated.confidence}"
        )
