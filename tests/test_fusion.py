import pytest
import numpy as np
from hypothesis import given, settings, strategies as st
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import CameraPose, Violation
from fusion.projector import project_to_3d, triangulate_violations, overlay_violations_3d

@st.composite
def camera_pose_generator(draw):
    return CameraPose(
        image_id="img1.jpg",
        position=draw(st.tuples(st.floats(-10, 10), st.floats(-10, 10), st.floats(-10, 10))),
        # simplified rotation quaternion
        rotation=(1.0, 0.0, 0.0, 0.0),
        confidence=1.0,
        intrinsics=np.array([800.0, 800.0, 400.0, 300.0])
    )

@settings(deadline=None)
@given(st.integers(0, 800), st.integers(0, 600), camera_pose_generator())
def test_2d_to_3d_projection_mathematics(u, v, pose):
    """Feature: construction-monitoring-system, Property 20: 2D-to-3D Projection Mathematics"""
    point_3d = project_to_3d((u, v), pose)
    
    assert len(point_3d) == 3
    assert isinstance(point_3d[0], float)
    assert isinstance(point_3d[1], float)
    assert isinstance(point_3d[2], float)

def test_multi_camera_triangulation():
    """Feature: construction-monitoring-system, Property 21: Multi-Camera Triangulation"""
    v1 = Violation(id="1", type="missing_helmet", worker_id="w1", severity="high", 
                   location_2d=(10, 10), location_3d=(1.0, 2.0, 3.0), confidence=0.9, timestamp=datetime.now())
    v2 = Violation(id="2", type="missing_helmet", worker_id="w1", severity="high", 
                   location_2d=(15, 15), location_3d=(3.0, 4.0, 5.0), confidence=0.8, timestamp=datetime.now())
    
    poses = {}
    triangulated = triangulate_violations([v1, v2], poses)
    
    assert len(triangulated) == 1
    t = triangulated[0]
    assert t.location_3d == (2.0, 3.0, 4.0)

def test_3d_violation_marker_overlay():
    """Feature: construction-monitoring-system, Property 22: 3D Violation Marker Overlay"""
    v1 = Violation(id="1", type="missing_helmet", worker_id="w1", severity="high", 
                   location_2d=(10, 10), location_3d=(1.0, 2.0, 3.0), confidence=0.9, timestamp=datetime.now())
    v2 = Violation(id="2", type="missing_vest", worker_id="w2", severity="medium", 
                   location_2d=(15, 15), location_3d=None, confidence=0.8, timestamp=datetime.now())
    
    markers = overlay_violations_3d(None, [v1, v2])
    
    assert len(markers["3d_markers"]) == 1
    assert len(markers["2d_fallback"]) == 1
    
    assert markers["3d_markers"][0]["color"] == "red"
