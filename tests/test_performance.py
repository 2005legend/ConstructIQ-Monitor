"""Performance property tests validating timing requirements."""
import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from progress.tracker import (
    analyze_progress,
    compare_images,
    calculate_zone_changes,
    generate_change_overlay,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_images_640x480():
    """Returns two 640×480 RGB images (one with a change block)."""
    img1 = np.full((480, 640, 3), 100, dtype=np.uint8)
    img2 = np.full((480, 640, 3), 100, dtype=np.uint8)
    img2[100:200, 100:200] = 200  # introduce a structural change
    return img1, img2


# ---------------------------------------------------------------------------
# Requirement 7.2 — analyze_progress must complete in < 1.0 s
# ---------------------------------------------------------------------------

def test_progress_module_timing_requirement_7_2(sample_images_640x480):
    """Requirement 7.2: analyze_progress on a 640×480 image pair must finish
    within 1.0 second.
    """
    img1, img2 = sample_images_640x480

    start = time.perf_counter()
    result = analyze_progress(img1, img2)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, (
        f"analyze_progress took {elapsed:.3f}s — exceeds 1.0 s requirement (Req 7.2)"
    )
    # Sanity-check the result while we're here
    assert 0.0 <= result.overall_change <= 100.0


# ---------------------------------------------------------------------------
# Sustained performance — 5 consecutive runs
# ---------------------------------------------------------------------------

def test_progress_module_timing_under_load(sample_images_640x480):
    """Average time over 5 consecutive analyze_progress calls must be < 1.0 s."""
    img1, img2 = sample_images_640x480
    times = []

    for _ in range(5):
        t0 = time.perf_counter()
        analyze_progress(img1, img2)
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    assert avg < 1.0, (
        f"Average analyze_progress time {avg:.3f}s exceeds 1.0 s requirement"
    )


# ---------------------------------------------------------------------------
# Individual pipeline stages
# ---------------------------------------------------------------------------

def test_change_overlay_timing(sample_images_640x480):
    """The compare → zone analysis → overlay pipeline must finish in < 1.0 s."""
    img1, img2 = sample_images_640x480

    start = time.perf_counter()

    diff_mask = compare_images(img1, img2)
    zones = calculate_zone_changes(diff_mask, grid_x=8, grid_y=8)
    overlay = generate_change_overlay(img2, diff_mask, zones)

    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, (
        f"Progress pipeline (compare+zones+overlay) took {elapsed:.3f}s — "
        "exceeds 1.0 s requirement"
    )
    assert overlay.shape == img1.shape, "Overlay shape must match input image shape"
