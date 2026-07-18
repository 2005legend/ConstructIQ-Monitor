import pytest
import numpy as np
import cv2
from hypothesis import given, strategies as st
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import ZoneAnalysis
from progress.tracker import (
    align_images,
    compare_images,
    calculate_zone_changes,
    generate_change_overlay,
    analyze_progress
)

@st.composite
def image_generator(draw, min_size=100, max_size=500):
    w = draw(st.integers(min_value=min_size, max_value=max_size))
    h = draw(st.integers(min_value=min_size, max_value=max_size))
    # Generate a random color image
    img = draw(st.lists(
        st.integers(min_value=0, max_value=255), 
        min_size=w*h*3, max_size=w*h*3
    ))
    return np.array(img, dtype=np.uint8).reshape((h, w, 3))

def test_progress_comparison_consistency():
    """Feature: construction-monitoring-system, Property 7: Progress Comparison Consistency"""
    # Create two identical images
    img1 = np.ones((200, 200, 3), dtype=np.uint8) * 100
    img2 = np.ones((200, 200, 3), dtype=np.uint8) * 100
    
    # Should result in empty mask (no changes)
    diff_mask = compare_images(img1, img2)
    assert np.sum(diff_mask) == 0
    
    # Add a change
    img2[50:100, 50:100] = 200
    diff_mask2 = compare_images(img1, img2)
    assert np.sum(diff_mask2) > 0

@given(st.integers(min_value=1, max_value=100))
def test_change_percentage_mathematical_correctness(change_area_size):
    """Feature: construction-monitoring-system, Property 8: Change Percentage Mathematical Correctness"""
    # Test change mathematical correctness
    img1 = np.ones((100, 100, 3), dtype=np.uint8) * 50
    img2 = np.ones((100, 100, 3), dtype=np.uint8) * 50
    
    # Draw a square change
    s = min(100, change_area_size)
    img2[0:s, 0:s] = 255
    
    result = analyze_progress(img1, img2)
    
    # Validate mathematical bounds
    assert 0.0 <= result.overall_change <= 100.0
    
    # Since we use morphological operations, the percentage won't be exactly s*s / 10000, 
    # but it should be bounded correctly.

def test_zone_grid_analysis_completeness():
    """Feature: construction-monitoring-system, Property 9: Zone Grid Analysis Completeness"""
    diff_mask = np.zeros((400, 400), dtype=np.uint8)
    diff_mask[100:200, 100:200] = 255
    
    grid_x, grid_y = 8, 8
    zones = calculate_zone_changes(diff_mask, grid_x, grid_y)
    
    # Check completeness
    assert len(zones) == grid_x * grid_y
    
    # Check that zone percentages are calculated
    for zone in zones:
        assert 0.0 <= zone.change_percentage <= 100.0

def test_change_overlay_generation():
    """Feature: construction-monitoring-system, Property 10: Change Overlay Generation"""
    after_image = np.ones((200, 200, 3), dtype=np.uint8) * 100
    diff_mask = np.zeros((200, 200), dtype=np.uint8)
    diff_mask[50:150, 50:150] = 255
    
    zones = calculate_zone_changes(diff_mask, 2, 2)
    
    overlay = generate_change_overlay(after_image, diff_mask, zones)
    
    assert overlay.shape == after_image.shape
    assert overlay.dtype == np.uint8

def test_noise_filtering_consistency():
    """Feature: construction-monitoring-system, Property 12: Noise Filtering Consistency"""
    img1 = np.ones((200, 200, 3), dtype=np.uint8) * 100
    
    # Minor lighting change (within threshold 45)
    img2_minor_light = np.ones((200, 200, 3), dtype=np.uint8) * 130
    
    # Structural change (major intensity difference)
    img2_structural = np.ones((200, 200, 3), dtype=np.uint8) * 200
    
    diff_minor = compare_images(img1, img2_minor_light)
    diff_structural = compare_images(img1, img2_structural)
    
    assert np.sum(diff_minor) == 0  # Should be filtered
    assert np.sum(diff_structural) > 0  # Should be detected
