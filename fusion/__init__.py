"""Fusion Engine - 2D-to-3D projection and violation spatial mapping."""
from .projector import project_to_3d, triangulate_violations, overlay_violations_3d, validate_3d_projection

__all__ = ["project_to_3d", "triangulate_violations", "overlay_violations_3d", "validate_3d_projection"]
