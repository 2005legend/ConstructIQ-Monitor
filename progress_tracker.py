import cv2
import numpy as np

def zone_analysis(diff_mask, grid=3):
    h, w = diff_mask.shape
    zones = {}
    for i in range(grid):
        for j in range(grid):
            zone = diff_mask[i*h//grid:(i+1)*h//grid, 
                           j*w//grid:(j+1)*w//grid]
            pct = (np.sum(zone > 0) / zone.size) * 100
            zones[f"Zone({i},{j})"] = round(pct, 2)
    return zones

def detect_progress_change(img1_array, img2_array):
    """
    Compare two images and detect structural changes.
    Args:
        img1_array, img2_array: numpy arrays of the two images.
    Returns:
        diff_image: Visualization of the change.
        change_percent: Percentage of pixels changed.
    """
    # Convert to grayscale
    if len(img1_array.shape) == 3:
        img1_gray = cv2.cvtColor(img1_array, cv2.COLOR_RGB2GRAY)
    else:
        img1_gray = img1_array
        
    if len(img2_array.shape) == 3:
        img2_gray = cv2.cvtColor(img2_array, cv2.COLOR_RGB2GRAY)
    else:
        img2_gray = img2_array
        
    # Resize img2 to match img1 dimensions to compute diff
    img2_gray = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))
    
    # Apply Gaussian blur to reduce noise
    img1_blur = cv2.GaussianBlur(img1_gray, (5, 5), 0)
    img2_blur = cv2.GaussianBlur(img2_gray, (5, 5), 0)
    
    # Compute absolute difference
    diff = cv2.absdiff(img1_blur, img2_blur)
    
    # Threshold the diff to create a binary mask
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Clean up the mask using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of changed regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate percentage change and zone analysis
    change_percent = (np.sum(thresh > 0) / thresh.size) * 100
    zones = zone_analysis(thresh, grid=3)
    
    # Draw contours on the second image for visualization
    # Resize original img2 color for overlay if necessary
    img2_color_resized = cv2.resize(img2_array, (img1_array.shape[1], img1_array.shape[0]))
    if len(img2_color_resized.shape) == 2:
        img2_color_resized = cv2.cvtColor(img2_color_resized, cv2.COLOR_GRAY2RGB)
        
    diff_image = img2_color_resized.copy()
    cv2.drawContours(diff_image, contours, -1, (255, 0, 0), 2) # Draw in red
    
    # Draw grid lines for zones
    h, w = diff_image.shape[:2]
    grid = 3
    for i in range(1, grid):
        cv2.line(diff_image, (0, i * h // grid), (w, i * h // grid), (0, 255, 0), 1)
        cv2.line(diff_image, (i * w // grid, 0), (i * w // grid, h), (0, 255, 0), 1)
    
    return diff_image, change_percent, zones
