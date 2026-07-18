"""Fusion Engine — 2D-to-3D projection and violation spatial mapping.

COLMAP convention:
  The stored translation vector `t` represents the world origin expressed in
  camera coordinates, i.e.  t = R @ C_world  where C_world is the camera
  centre in world space.

  Camera-to-world transform:
      P_world = R.T @ P_camera + C_world
             = R.T @ P_camera - R.T @ t
"""
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import Violation, CameraPose, Detection


def project_to_3d(
    detection_2d: Tuple[int, int],
    pose: CameraPose,
    depth: float = 5.0,
) -> Tuple[float, float, float]:
    """Projects a 2D pixel coordinate to a 3D world point.

    Uses the pinhole camera model and the COLMAP camera-to-world transform:
        P_world = R.T @ P_camera - R.T @ t

    where ``t`` is stored in ``pose.position`` and is the translation of the
    world origin *in the camera frame* (COLMAP convention).

    Args:
        detection_2d: (u, v) pixel coordinate.
        pose: Camera pose containing COLMAP-style extrinsics and intrinsics.
        depth: Assumed scene depth in metres (used when no depth map is
            available).

    Returns:
        (X, Y, Z) 3D world coordinates as a tuple of floats.
    """
    u, v = detection_2d

    # --- Intrinsics -------------------------------------------------------
    if pose.intrinsics is not None and len(pose.intrinsics) >= 4:
        fx, fy = float(pose.intrinsics[0]), float(pose.intrinsics[1])
        cx, cy = float(pose.intrinsics[2]), float(pose.intrinsics[3])
    else:
        fx, fy = 800.0, 800.0
        cx, cy = 400.0, 300.0

    # --- Back-project into camera frame -----------------------------------
    x_n = (u - cx) / fx   # normalised x
    y_n = (v - cy) / fy   # normalised y
    # Scale to the assumed depth
    point_camera = np.array([x_n * depth, y_n * depth, float(depth)])

    # --- Rotation matrix from quaternion ----------------------------------
    from scipy.spatial.transform import Rotation

    q = np.asarray(pose.rotation, dtype=float)

    if len(q) == 4:
        # COLMAP stores quaternions in wxyz order.
        # scipy.spatial.transform.Rotation.from_quat expects xyzw order.
        # Detect order: if first component looks like the scalar (close to 1
        # for small rotations / identity) we treat it as wxyz and reorder.
        # Heuristic: COLMAP wxyz → scipy xyzw means swapping [w,x,y,z] to
        # [x,y,z,w].
        q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    else:
        # Unexpected length — fall back to identity
        q_xyzw = np.array([0.0, 0.0, 0.0, 1.0])

    # Normalise to be safe
    norm = np.linalg.norm(q_xyzw)
    if norm > 1e-8:
        q_xyzw = q_xyzw / norm

    r = Rotation.from_quat(q_xyzw)
    R = r.as_matrix()  # shape (3, 3)

    # --- COLMAP camera-to-world transform ---------------------------------
    # t = pose.position is the translation stored in COLMAP (world origin in
    # camera frame).
    t = np.asarray(pose.position, dtype=float)

    # Camera centre in world: C = -R.T @ t
    # World point:  P_world = R.T @ P_camera + C
    #                       = R.T @ P_camera - R.T @ t
    point_world = R.T @ point_camera - R.T @ t

    return float(point_world[0]), float(point_world[1]), float(point_world[2])


def validate_3d_projection(
    detection_2d: Tuple[int, int],
    pose: CameraPose,
    depth: float = 5.0,
) -> float:
    """Validates a 3D projection and returns a confidence score.

    Checks whether the point has a positive depth in the camera frame
    (i.e. it is in front of the camera) and returns a confidence value
    between 0.0 and 1.0.

    Args:
        detection_2d: (u, v) pixel coordinate.
        pose: Camera pose.
        depth: Assumed depth used for the projection.

    Returns:
        Confidence score in [0.0, 1.0].  Returns 0.0 when the projected
        depth is non-positive (point behind the camera).
    """
    u, v = detection_2d

    if pose.intrinsics is not None and len(pose.intrinsics) >= 4:
        fx, fy = float(pose.intrinsics[0]), float(pose.intrinsics[1])
        cx, cy = float(pose.intrinsics[2]), float(pose.intrinsics[3])
    else:
        fx, fy = 800.0, 800.0
        cx, cy = 400.0, 300.0

    x_n = (u - cx) / fx
    y_n = (v - cy) / fy
    z_camera = float(depth)

    # A valid projection requires the point to be in front of the camera.
    if z_camera <= 0.0:
        return 0.0

    # Confidence based on how "central" the ray is (lower distortion near
    # principal point → higher confidence).
    ray_magnitude = float(np.sqrt(x_n ** 2 + y_n ** 2 + 1.0))
    # Normalised confidence: 1/ray_magnitude is highest at the principal point
    confidence = float(np.clip(1.0 / ray_magnitude, 0.0, 1.0))
    return confidence


def triangulate_violations(
    violations: List[Violation],
    poses_dict: Dict[str, CameraPose],
) -> List[Violation]:
    """Improves 3D accuracy by averaging multi-view projections.

    Groups violations by (worker_id, type) and averages the 3D positions
    when multiple estimates exist.  A full DLT implementation would be used
    in production.
    """
    grouped: Dict[tuple, List[Violation]] = {}
    for v in violations:
        key = (v.worker_id, v.type)
        grouped.setdefault(key, []).append(v)

    triangulated: List[Violation] = []

    for key, group in grouped.items():
        if len(group) == 1:
            triangulated.append(group[0])
            continue

        valid_3d = [v.location_3d for v in group if v.location_3d is not None]
        if valid_3d:
            avg_x = sum(p[0] for p in valid_3d) / len(valid_3d)
            avg_y = sum(p[1] for p in valid_3d) / len(valid_3d)
            avg_z = sum(p[2] for p in valid_3d) / len(valid_3d)
            merged = group[0]
            merged.location_3d = (avg_x, avg_y, avg_z)
            triangulated.append(merged)
        else:
            triangulated.append(group[0])

    return triangulated


def overlay_violations_3d(
    point_cloud_data: Any,
    violations: List[Violation],
) -> dict:
    """Generates visualisation markers for 3D display.

    Returns a dictionary of markers suitable for Plotly or similar.
    Falls back to 2D if ``location_3d`` is ``None``.
    """
    markers: Dict[str, list] = {
        "3d_markers": [],
        "2d_fallback": [],
    }

    for v in violations:
        if v.location_3d is not None:
            markers["3d_markers"].append({
                "id": v.id,
                "type": v.type,
                "position": v.location_3d,
                "severity": v.severity,
                "color": "red" if v.severity == "high" else "orange",
            })
        else:
            markers["2d_fallback"].append({
                "id": v.id,
                "type": v.type,
                "position_2d": v.location_2d,
                "severity": v.severity,
            })

    return markers
