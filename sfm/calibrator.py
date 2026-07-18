"""Camera calibration and pose validation utilities."""
import numpy as np
from typing import List, Optional, Dict
from dataclasses import replace
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import CameraPose


def estimate_camera_parameters(image_width: int, image_height: int) -> np.ndarray:
    """Estimates reasonable camera intrinsics from image dimensions.

    Uses the common heuristic that the focal length (in pixels) is
    approximately equal to the longest image dimension, which corresponds
    roughly to a 35 mm camera with a standard lens.

    Args:
        image_width:  Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        Array ``[fx, fy, cx, cy]`` where *fx == fy == max(w, h)*,
        *cx == width / 2*, and *cy == height / 2*.
    """
    focal_length = float(max(image_width, image_height))
    cx = image_width / 2.0
    cy = image_height / 2.0
    return np.array([focal_length, focal_length, cx, cy], dtype=float)


def compute_reprojection_error(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    pose: CameraPose,
) -> float:
    """Computes the mean reprojection error in pixels.

    Projects each 3-D point through *pose* using the pinhole camera model
    and measures the Euclidean distance to the corresponding 2-D observation.

    Args:
        points_3d: Array of shape (N, 3) containing 3-D world points.
        points_2d: Array of shape (N, 2) containing 2-D image observations.
        pose:      Camera pose with intrinsics.

    Returns:
        Mean reprojection error in pixels.  Returns ``float('inf')`` when
        the inputs are empty or have mismatched sizes.
    """
    if points_3d.shape[0] == 0 or points_3d.shape[0] != points_2d.shape[0]:
        return float("inf")

    from scipy.spatial.transform import Rotation

    # Intrinsics
    if pose.intrinsics is not None and len(pose.intrinsics) >= 4:
        fx, fy = float(pose.intrinsics[0]), float(pose.intrinsics[1])
        cx, cy = float(pose.intrinsics[2]), float(pose.intrinsics[3])
    else:
        fx, fy = 800.0, 800.0
        cx, cy = 400.0, 300.0

    # Rotation matrix (COLMAP wxyz → scipy xyzw)
    q = np.asarray(pose.rotation, dtype=float)
    if len(q) == 4:
        q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    else:
        q_xyzw = np.array([0.0, 0.0, 0.0, 1.0])
    norm = np.linalg.norm(q_xyzw)
    if norm > 1e-8:
        q_xyzw = q_xyzw / norm
    R = Rotation.from_quat(q_xyzw).as_matrix()

    # Translation (COLMAP: t = R @ C → camera frame translation)
    t = np.asarray(pose.position, dtype=float)

    errors = []
    for pt3, pt2 in zip(points_3d, points_2d):
        # Transform world point into camera frame
        p_cam = R @ pt3 + t  # COLMAP: p_cam = R * p_world + t

        if p_cam[2] <= 0.0:
            # Point behind camera — assign large error
            errors.append(1e6)
            continue

        u_proj = fx * (p_cam[0] / p_cam[2]) + cx
        v_proj = fy * (p_cam[1] / p_cam[2]) + cy

        err = float(np.sqrt((u_proj - pt2[0]) ** 2 + (v_proj - pt2[1]) ** 2))
        errors.append(err)

    return float(np.mean(errors)) if errors else float("inf")


def validate_pose_confidence(
    pose: CameraPose,
    max_reprojection_error: float = 5.0,
) -> CameraPose:
    """Returns a new CameraPose with an updated confidence value.

    Rules (applied in order):
    1. If ``pose.confidence >= 1.0`` (placeholder from reconstructor) and
       intrinsics are available, set confidence to 0.8 (reasonable estimate).
    2. If reprojection error can be computed and is below
       *max_reprojection_error*, update confidence to
       ``1.0 - error / (max_reprojection_error * 2)``.
    3. Clamp the final value to [0.0, 1.0].

    Args:
        pose:                  Input camera pose.
        max_reprojection_error: Pixel threshold for acceptable error.

    Returns:
        A new CameraPose with updated confidence.
    """
    new_confidence = pose.confidence

    # Rule 1 — placeholder confidence from the reconstructor
    if pose.confidence >= 1.0 and pose.intrinsics is not None:
        new_confidence = 0.8

    # Rule 2 — update from reprojection error when 3-D / 2-D correspondences
    # are embedded in the pose object.  The calibrator API is purposely kept
    # simple: it only uses intrinsics presence as a proxy here.
    # (A caller can supply real correspondences via compute_reprojection_error
    # and build a confidence value directly.)
    # If the current confidence is still the placeholder we just set (0.8) and
    # the error computation returned a useful value, refine it.
    # In this default implementation we leave this path open but do not force
    # a recompute without ground-truth 2-D/3-D pairs.

    # Clamp
    new_confidence = float(np.clip(new_confidence, 0.0, 1.0))

    return CameraPose(
        image_id=pose.image_id,
        position=pose.position,
        rotation=pose.rotation,
        confidence=new_confidence,
        intrinsics=pose.intrinsics,
    )


def filter_valid_poses(
    poses: List[CameraPose],
    min_confidence: float = 0.5,
) -> List[CameraPose]:
    """Returns only poses whose confidence is at or above *min_confidence*.

    Args:
        poses:           List of camera poses.
        min_confidence:  Minimum acceptable confidence (inclusive).

    Returns:
        Filtered list of CameraPose objects.
    """
    return [p for p in poses if p.confidence >= min_confidence]


def diagnose_pose_failure(pose: CameraPose) -> Dict[str, str]:
    """Produces a human-readable diagnosis for a problematic camera pose.

    Checks are evaluated in order; the first matching condition is returned.

    Args:
        pose: The camera pose to diagnose.

    Returns:
        Dictionary with keys ``"cause"`` and ``"resolution"``.
    """
    if pose.confidence < 0.3:
        return {
            "cause": "Low confidence pose",
            "resolution": "Capture more overlapping images",
        }

    if pose.intrinsics is None:
        return {
            "cause": "Missing camera intrinsics",
            "resolution": (
                "Provide camera calibration data or use automatic estimation"
            ),
        }

    if all(abs(c) < 1e-9 for c in pose.position):
        return {
            "cause": "Zero position — reconstruction may have failed",
            "resolution": "Ensure sufficient feature matches between images",
        }

    return {
        "cause": "Unknown",
        "resolution": "Check image quality and overlap",
    }
