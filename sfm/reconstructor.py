"""
SfM Module — OpenCV-based Structure-from-Motion pipeline.

Uses SIFT feature matching + Essential Matrix decomposition + incremental
triangulation.  Does NOT require COLMAP or pycolmap binaries.
"""
import os
import struct
import cv2
import numpy as np
from typing import List, Optional, Any
from datetime import datetime
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import CameraPose, ReconstructionResult


# ---------------------------------------------------------------------------
# Intrinsics estimation
# ---------------------------------------------------------------------------

def _estimate_K(img: np.ndarray) -> np.ndarray:
    """Estimate a reasonable camera intrinsic matrix from image dimensions."""
    h, w = img.shape[:2]
    f = max(w, h)          # focal length heuristic (35mm equiv)
    cx, cy = w / 2.0, h / 2.0
    return np.array([[f, 0, cx],
                     [0, f, cy],
                     [0, 0,  1]], dtype=np.float64)


# ---------------------------------------------------------------------------
# Feature matching
# ---------------------------------------------------------------------------

def _match_features(img1: np.ndarray, img2: np.ndarray):
    """SIFT + FLANN feature matching with Lowe ratio test.

    Returns (src_pts, dst_pts) as float32 arrays of shape (N, 2), or
    raises ValueError if too few matches are found.
    """
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(nfeatures=5000)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        raise ValueError("Not enough keypoints detected.")

    index_params = dict(algorithm=1, trees=5)   # FLANN KD-tree
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw_matches = flann.knnMatch(des1, des2, k=2)

    good = [m for m, n in raw_matches if m.distance < 0.75 * n.distance]

    if len(good) < 10:
        raise ValueError(
            f"Only {len(good)} good matches — images may not overlap sufficiently."
        )

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
    return src_pts, dst_pts, kp1, kp2, good


# ---------------------------------------------------------------------------
# Pose recovery
# ---------------------------------------------------------------------------

def _recover_pose(src_pts, dst_pts, K):
    """Compute Essential Matrix and recover R, t."""
    E, mask = cv2.findEssentialMat(
        src_pts, dst_pts, K,
        method=cv2.RANSAC, prob=0.999, threshold=1.0,
    )
    if E is None:
        raise ValueError("Essential matrix computation failed.")

    _, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, K, mask=mask)
    inlier_count = int(mask_pose.sum()) if mask_pose is not None else len(src_pts)
    return R, t, inlier_count


# ---------------------------------------------------------------------------
# Triangulation
# ---------------------------------------------------------------------------

def _triangulate(src_pts, dst_pts, P1, P2):
    """Triangulate points and return (N, 3) float array."""
    pts4d = cv2.triangulatePoints(P1, P2, src_pts.T, dst_pts.T)
    w = pts4d[3]
    # Avoid division by near-zero
    valid = np.abs(w) > 1e-7
    pts3d = np.full((4, pts4d.shape[1]), np.nan)
    pts3d[:, valid] = pts4d[:, valid] / w[valid]
    return pts3d[:3].T   # (N, 3)


# ---------------------------------------------------------------------------
# Color extraction
# ---------------------------------------------------------------------------

def _extract_colors(img_bgr: np.ndarray, pts2d: np.ndarray) -> np.ndarray:
    """Sample per-point BGR colors from image, clamp to valid bounds."""
    h, w = img_bgr.shape[:2]
    colors = np.zeros((len(pts2d), 3), dtype=np.uint8)
    for i, (x, y) in enumerate(pts2d):
        xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
        b, g, r = img_bgr[yi, xi]
        colors[i] = [r, g, b]   # store as RGB
    return colors


# ---------------------------------------------------------------------------
# PLY writer (avoids pycolmap completely)
# ---------------------------------------------------------------------------

def _write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Write a binary little-endian PLY file with xyz + rgb."""
    n = len(xyz)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    with open(path, "wb") as f:
        f.write(header)
        for i in range(n):
            x, y, z = float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2])
            r, g, b = int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])
            f.write(struct.pack("<fffBBB", x, y, z, r, g, b))


# ---------------------------------------------------------------------------
# Outlier filtering
# ---------------------------------------------------------------------------

def _filter_outliers(xyz: np.ndarray, rgb: np.ndarray, sigma: float = 3.0):
    """Remove statistical outliers and NaN/Inf points."""
    # Drop NaN / Inf
    finite = np.all(np.isfinite(xyz), axis=1)
    xyz, rgb = xyz[finite], rgb[finite]

    if len(xyz) == 0:
        return xyz, rgb

    # Sigma-based filter per axis
    med = np.median(xyz, axis=0)
    std = xyz.std(axis=0) + 1e-9
    inlier = np.all(np.abs(xyz - med) < sigma * std, axis=1)
    return xyz[inlier], rgb[inlier]


# ---------------------------------------------------------------------------
# Main reconstruction entry point
# ---------------------------------------------------------------------------

def reconstruct_3d(image_paths: List[str], output_path: str) -> ReconstructionResult:
    """Incremental SfM using OpenCV SIFT + Essential Matrix + Triangulation.

    Works entirely with OpenCV — no COLMAP binary or pycolmap required.

    Args:
        image_paths: List of absolute paths to input images (≥3 recommended,
                     8-15 ideal, all of the same scene with ≥60 % overlap).
        output_path: Directory where point_cloud.ply will be saved.

    Returns:
        ReconstructionResult with success flag, camera poses, PLY path and
        quality metrics.
    """
    start = datetime.now()

    if len(image_paths) < 2:
        return ReconstructionResult(
            success=False,
            camera_poses=[],
            point_cloud_path="",
            quality_score=0.0,
            error_message="Need at least 2 images; 8-15 overlapping images recommended.",
            processing_time=0.0,
        )

    # --- Load images -------------------------------------------------------
    images = []
    valid_paths = []
    for p in image_paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)
            valid_paths.append(p)

    if len(images) < 2:
        return ReconstructionResult(
            success=False,
            camera_poses=[],
            point_cloud_path="",
            quality_score=0.0,
            error_message="Could not load images. Check file paths and formats.",
            processing_time=(datetime.now() - start).total_seconds(),
        )

    # Use intrinsics estimated from the first image (assumes same camera)
    K = _estimate_K(images[0])

    # --- Incremental reconstruction ----------------------------------------
    all_xyz: List[np.ndarray] = []
    all_rgb: List[np.ndarray] = []
    camera_poses: List[CameraPose] = []
    total_inliers = 0
    failed_pairs = 0

    # Anchor: first camera at identity
    R_prev = np.eye(3, dtype=np.float64)
    t_prev = np.zeros((3, 1), dtype=np.float64)

    camera_poses.append(CameraPose(
        image_id=os.path.basename(valid_paths[0]),
        position=(0.0, 0.0, 0.0),
        rotation=(1.0, 0.0, 0.0, 0.0),
        confidence=1.0,
        intrinsics=np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]),
    ))

    P_prev = K @ np.hstack([R_prev, t_prev])   # projection matrix of frame 0

    for i in range(len(images) - 1):
        img1, img2 = images[i], images[i + 1]

        try:
            src_pts, dst_pts, kp1, kp2, good = _match_features(img1, img2)
        except ValueError as e:
            print(f"[SfM] Pair {i}-{i+1} matching failed: {e}")
            failed_pairs += 1
            # Advance pose chain with identity step so later pairs still chain
            camera_poses.append(CameraPose(
                image_id=os.path.basename(valid_paths[i + 1]),
                position=tuple((-R_prev.T @ t_prev).flatten()),
                rotation=(1.0, 0.0, 0.0, 0.0),
                confidence=0.3,
                intrinsics=np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]),
            ))
            continue

        try:
            R_rel, t_rel, inliers = _recover_pose(src_pts, dst_pts, K)
        except ValueError as e:
            print(f"[SfM] Pair {i}-{i+1} pose recovery failed: {e}")
            failed_pairs += 1
            continue

        total_inliers += inliers

        # Normalise translation (scale ambiguity in monocular SfM)
        t_norm = np.linalg.norm(t_rel)
        if t_norm > 1e-8:
            t_rel = t_rel / t_norm

        # Chain global pose
        R_cur = R_rel @ R_prev
        t_cur = t_prev + R_prev @ t_rel

        # Camera centre in world coords: C = -R^T t
        C_world = (-R_cur.T @ t_cur).flatten()
        # Rotation as wxyz quaternion from rotation matrix
        q = _rotation_matrix_to_wxyz_quaternion(R_cur)

        camera_poses.append(CameraPose(
            image_id=os.path.basename(valid_paths[i + 1]),
            position=(float(C_world[0]), float(C_world[1]), float(C_world[2])),
            rotation=(float(q[0]), float(q[1]), float(q[2]), float(q[3])),
            confidence=min(1.0, inliers / max(len(src_pts), 1)),
            intrinsics=np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]]),
        ))

        # Triangulate
        P_cur = K @ np.hstack([R_cur, t_cur])
        pts3d = _triangulate(src_pts, dst_pts, P_prev, P_cur)

        # Sample colors from img1 at src_pts
        colors = _extract_colors(img1, src_pts)

        # Keep only points with positive depth in both cameras
        in_front = _cheirality_mask(pts3d, R_prev, t_prev, R_cur, t_cur)
        pts3d = pts3d[in_front]
        colors = colors[in_front]

        all_xyz.append(pts3d)
        all_rgb.append(colors)

        # Advance
        R_prev, t_prev, P_prev = R_cur, t_cur, P_cur

    elapsed = (datetime.now() - start).total_seconds()

    if not all_xyz:
        return ReconstructionResult(
            success=False,
            camera_poses=camera_poses,
            point_cloud_path="",
            quality_score=0.0,
            error_message=(
                "Reconstruction failed — no image pairs had enough matching features. "
                "Ensure images overlap by at least 60% and show rich texture."
            ),
            processing_time=elapsed,
        )

    # --- Build and clean point cloud ---------------------------------------
    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)
    xyz, rgb = _filter_outliers(xyz, rgb, sigma=3.0)

    if len(xyz) == 0:
        return ReconstructionResult(
            success=False,
            camera_poses=camera_poses,
            point_cloud_path="",
            quality_score=0.0,
            error_message=(
                "Reconstruction produced only outlier points. "
                "Try images with more texture and overlap."
            ),
            processing_time=elapsed,
        )

    # --- Write PLY ---------------------------------------------------------
    os.makedirs(output_path, exist_ok=True)
    ply_path = os.path.join(output_path, "point_cloud.ply")
    _write_ply(ply_path, xyz, rgb)

    # Quality score: ratio of good pairs + inlier density
    n_pairs = len(images) - 1
    good_pair_ratio = (n_pairs - failed_pairs) / max(n_pairs, 1)
    avg_inliers = total_inliers / max(n_pairs - failed_pairs, 1)
    inlier_score = min(1.0, avg_inliers / 200.0)   # 200 inliers = perfect
    quality = round(0.5 * good_pair_ratio + 0.5 * inlier_score, 3)

    return ReconstructionResult(
        success=True,
        camera_poses=camera_poses,
        point_cloud_path=ply_path,
        quality_score=quality,
        error_message=None,
        processing_time=elapsed,
    )


# ---------------------------------------------------------------------------
# Helper: cheirality mask
# ---------------------------------------------------------------------------

def _cheirality_mask(
    pts3d: np.ndarray,
    R1: np.ndarray, t1: np.ndarray,
    R2: np.ndarray, t2: np.ndarray,
) -> np.ndarray:
    """Return boolean mask where points have positive depth in BOTH cameras."""
    # Depth in camera 1: z = R1[2] @ X + t1[2]
    depth1 = (R1[2] @ pts3d.T + t1[2]).flatten()
    # Depth in camera 2
    depth2 = (R2[2] @ pts3d.T + t2[2]).flatten()
    return (depth1 > 0) & (depth2 > 0) & np.all(np.isfinite(pts3d), axis=1)


# ---------------------------------------------------------------------------
# Helper: rotation matrix → wxyz quaternion
# ---------------------------------------------------------------------------

def _rotation_matrix_to_wxyz_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to [w, x, y, z] unit quaternion."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / (np.linalg.norm(q) + 1e-12)


# ---------------------------------------------------------------------------
# Kept for backward compatibility with tests / __init__.py
# ---------------------------------------------------------------------------

def extract_camera_poses(reconstruction: Any) -> List[CameraPose]:
    """Passthrough for test compatibility — reconstruction is already a list."""
    if isinstance(reconstruction, list):
        return reconstruction
    return []


def generate_point_cloud(reconstruction: Any, output_ply: str) -> None:
    """Passthrough for test compatibility."""
    pass
