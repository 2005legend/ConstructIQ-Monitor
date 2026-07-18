import pytest
import os
import tempfile
from hypothesis import given, strategies as st
from unittest.mock import patch, MagicMock
import sys

# Mock pycolmap before it gets imported by sfm.reconstructor
sys.modules['pycolmap'] = MagicMock()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sfm.reconstructor import reconstruct_3d, extract_camera_poses, generate_point_cloud

from models import CameraPose, ReconstructionResult

def test_sfm_reconstruction_attempt_insufficient_images():
    """Feature: construction-monitoring-system, Property 13: SfM Reconstruction Attempt"""
    # Attempt with 1 image (insufficient)
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = reconstruct_3d(["image1.jpg"], tmp_dir)
        assert result.success is False
        assert "Insufficient images" in result.error_message

@patch("sfm.reconstructor.pycolmap")
def test_sfm_reconstruction_attempt(mock_pycolmap):
    """Feature: construction-monitoring-system, Property 13: SfM Reconstruction Attempt (Mocked)"""
    mock_pycolmap.incremental_mapping.return_value = {"0": MagicMock()}
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = reconstruct_3d(["img1.jpg", "img2.jpg", "img3.jpg"], tmp_dir)
        # Even with mocked, we just check if it calls COLMAP functions
        assert mock_pycolmap.extract_features.called
        assert mock_pycolmap.match_exhaustive.called
        assert mock_pycolmap.incremental_mapping.called

@patch("sfm.reconstructor.pycolmap")
def test_reconstruction_error_reporting(mock_pycolmap):
    """Feature: construction-monitoring-system, Property 16: Reconstruction Error Reporting"""
    mock_pycolmap.incremental_mapping.return_value = {} # No maps created
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = reconstruct_3d(["img1.jpg", "img2.jpg"], tmp_dir)
        assert result.success is False
        assert "insufficient feature matches" in result.error_message.lower()

# Mock objects for camera pose testing
class MockImage:
    def __init__(self, name):
        self.name = name
        self.camera_id = 1
        self.tvec = [0.1, 0.2, 0.3]
        self.qvec = [1.0, 0.0, 0.0, 0.0]

class MockCamera:
    def __init__(self):
        self.params = [800, 400, 300] # focal length, cx, cy

class MockReconstruction:
    def __init__(self):
        self.images = {1: MockImage("img1.jpg"), 2: MockImage("img2.jpg")}
        self.cameras = {1: MockCamera()}
        
    def export_PLY(self, path):
        with open(path, 'w') as f:
            f.write("ply")

def test_camera_pose_recovery():
    """Feature: construction-monitoring-system, Property 14: Camera Pose Recovery"""
    mock_rec = MockReconstruction()
    poses = extract_camera_poses(mock_rec)
    
    assert len(poses) == 2
    for pose in poses:
        assert isinstance(pose, CameraPose)
        assert len(pose.position) == 3
        assert len(pose.rotation) == 4

def test_point_cloud_generation_format():
    """Feature: construction-monitoring-system, Property 15: Point Cloud Generation Format"""
    mock_rec = MockReconstruction()
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        ply_path = os.path.join(tmp_dir, "test.ply")
        generate_point_cloud(mock_rec, ply_path)
        
        assert os.path.exists(ply_path)
        with open(ply_path, 'r') as f:
            content = f.read()
            assert "ply" in content
