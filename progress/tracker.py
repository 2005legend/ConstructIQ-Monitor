import cv2
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict
import sys
import os

# Add parent directory to path to import models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import ZoneAnalysis, ProgressResult

def align_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Aligns img2 to match img1 using ORB feature matching and homography.
    If alignment fails, it falls back to simple resizing.
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
    
    # Initialize ORB
    orb = cv2.ORB_create(nfeatures=5000)
    
    # Detect features and compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
    
    if descriptors1 is None or descriptors2 is None:
        return cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
    # Match features
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    
    # Keep top 10%
    num_good_matches = int(len(matches) * 0.1)
    if num_good_matches < 10:
        return cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
    good_matches = matches[:num_good_matches]
    
    # Extract location of good matches
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)
    
    for i, match in enumerate(good_matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
        
    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    
    if h is None:
        return cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
    # Warp img2
    height, width = img1.shape[:2]
    img2_aligned = cv2.warpPerspective(img2, h, (width, height))
    
    return img2_aligned

def compare_images(before_image: np.ndarray, after_image: np.ndarray) -> np.ndarray:
    """
    Pixel-level comparison between two aligned images, filtering noise.
    Returns a binary mask of changes.
    """
    gray1 = cv2.cvtColor(before_image, cv2.COLOR_RGB2GRAY) if len(before_image.shape) == 3 else before_image
    gray2 = cv2.cvtColor(after_image, cv2.COLOR_RGB2GRAY) if len(after_image.shape) == 3 else after_image
    
    # Apply Gaussian blur to reduce noise
    blur1 = cv2.GaussianBlur(gray1, (5, 5), 0)
    blur2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    
    # Compute absolute difference
    diff = cv2.absdiff(blur1, blur2)
    
    # Noise filtering for lighting variations using adaptive thresholding or Otsu
    # For construction site changes, we use a fixed threshold to filter minor lighting changes
    _, thresh = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)
    
    # Morphological operations to clean up
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh

def calculate_zone_changes(diff_mask: np.ndarray, grid_x: int = 8, grid_y: int = 8) -> List[ZoneAnalysis]:
    """
    Divides image into grid zones and calculates change per zone.
    """
    h, w = diff_mask.shape
    zones = []
    
    step_y = h // grid_y
    step_x = w // grid_x
    
    for i in range(grid_y):
        for j in range(grid_x):
            y1, y2 = i * step_y, min((i + 1) * step_y, h)
            x1, x2 = j * step_x, min((j + 1) * step_x, w)
            
            zone = diff_mask[y1:y2, x1:x2]
            pct = (np.sum(zone > 0) / zone.size) * 100 if zone.size > 0 else 0
            
            # Categorize change type
            change_type = "no_change"
            if pct > 15:
                change_type = "major_change"
            elif pct > 5:
                change_type = "minor_change"
                
            zone_id = f"{chr(65+i)}{j+1}"  # E.g., A1, B2
            
            zones.append(ZoneAnalysis(
                zone_id=zone_id,
                coordinates=(x1, y1, x2, y2),
                change_percentage=round(pct, 2),
                change_type=change_type
            ))
            
    return zones

def generate_change_overlay(after_image: np.ndarray, diff_mask: np.ndarray, zones: List[ZoneAnalysis]) -> np.ndarray:
    """
    Creates colored overlays highlighting changed areas and zone grid.
    """
    overlay = after_image.copy()
    
    # Create colored mask for changes (Red)
    color_mask = np.zeros_like(overlay)
    color_mask[diff_mask > 0] = [255, 0, 0]  # Assuming RGB
    
    # Blend
    alpha = 0.5
    cv2.addWeighted(color_mask, alpha, overlay, 1 - alpha, 0, overlay)
    
    # Draw contours around changes for clarity
    contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 50, 50), 2)
    
    # Draw grid
    for zone in zones:
        x1, y1, x2, y2 = zone.coordinates
        # Draw grid lines
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Add text for zones with change
        if zone.change_percentage > 5:
            text = f"{zone.zone_id}: {zone.change_percentage}%"
            cv2.putText(overlay, text, (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
    return overlay

def analyze_progress(img1: np.ndarray, img2: np.ndarray, grid_x: int = 8, grid_y: int = 8) -> ProgressResult:
    """
    Main function to run progress tracking workflow.
    """
    img2_aligned = align_images(img1, img2)
    diff_mask = compare_images(img1, img2_aligned)
    zones = calculate_zone_changes(diff_mask, grid_x, grid_y)
    
    overall_change = (np.sum(diff_mask > 0) / diff_mask.size) * 100
    
    overlay = generate_change_overlay(img2_aligned, diff_mask, zones)
    
    return ProgressResult(
        overall_change=round(overall_change, 2),
        zone_analysis=zones,
        before_image=img1,
        after_image=img2_aligned,
        change_overlay=overlay,
        timestamp=datetime.now()
    )

# --- 3D Progress Tracking (ICP) ---

def icp(src: np.ndarray, dst: np.ndarray, max_iterations: int = 50, tolerance: float = 1e-5):
    """
    Simple point-to-point ICP using numpy and scipy KDTree.
    Returns (aligned_src, transformation_matrix).
    """
    from scipy.spatial import KDTree
    
    src = np.asarray(src)
    dst = np.asarray(dst)
    
    if len(src) == 0 or len(dst) == 0:
        return src, np.eye(4)
        
    T = np.eye(4)
    tree = KDTree(dst)
    
    prev_error = float('inf')
    src_aligned = src.copy()
    
    for i in range(max_iterations):
        distances, indices = tree.query(src_aligned)
        
        # Filter outliers
        valid = distances < np.percentile(distances, 80)
        
        A = src_aligned[valid]
        B = dst[indices[valid]]
        
        if len(A) < 10:
            break
            
        # Centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        
        # Center points
        AA = A - centroid_A
        BB = B - centroid_B
        
        # SVD
        H = AA.T @ BB
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Reflection fix
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
            
        t = centroid_B - R @ centroid_A
        
        # Update src_aligned
        src_aligned = (R @ src_aligned.T).T + t
        
        # Update transform matrix
        T_mat = np.eye(4)
        T_mat[:3, :3] = R
        T_mat[:3, 3] = t
        T = T_mat @ T
        
        # Error
        mean_error = np.mean(distances[valid])
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
        
    return src_aligned, T

def compare_point_clouds(xyz_past: np.ndarray, xyz_current: np.ndarray, threshold: float = 0.5):
    """
    Aligns xyz_current to xyz_past using ICP, then finds points in xyz_current 
    that are further than `threshold` from xyz_past.
    These new points represent volumetric progress/additions.
    """
    from scipy.spatial import KDTree
    
    if len(xyz_past) == 0 or len(xyz_current) == 0:
        return xyz_current, np.empty((0,3)), 0.0
        
    # Align current cloud to past cloud
    aligned_current, _ = icp(xyz_current, xyz_past)
    
    # Find points in the aligned current cloud that have no neighbors in the past cloud
    tree = KDTree(xyz_past)
    distances, _ = tree.query(aligned_current)
    
    added_mask = distances > threshold
    added_points = aligned_current[added_mask]
    
    percent_changed = (len(added_points) / len(aligned_current)) * 100 if len(aligned_current) > 0 else 0
    
    return aligned_current, added_points, round(percent_changed, 2)
