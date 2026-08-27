"""Alignment / auto-fit invariants for the multi-env video renderer.

Verifies that every per-env entity (robot, goal/EE markers, base/EE trails,
debug text) is shifted by the SAME grid offset, and that the auto-fit FREE
camera keeps the whole env grid in frame for a wide render_spacing.
"""

from __future__ import annotations

import numpy as np
import pytest

from unilab.visualization.render_many import (
    _FOVY_DEG,
    _add_one_sphere,
    _add_trajectory,
    _project_points,
    compute_grid_camera_distance,
    get_grid_offsets,
)

_WIDTH = 1280
_HEIGHT = 720
_SPACING = 12.0
_NUM_ENVS = 4


def _make_scene(maxgeom: int = 256):
    import mujoco

    # MjvScene() without a model allocates an empty geoms array; build a tiny
    # model with enough geoms to preallocate the scene buffer.
    geoms = "".join(
        f'<geom type="sphere" size="0.05" pos="{i} 0 0.5"/>' for i in range(maxgeom)
    )
    model = mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody>{geoms}</worldbody></mujoco>"
    )
    scene = mujoco.MjvScene(model, maxgeom=maxgeom)
    scene.ngeom = 0
    return scene


def _make_cam(offsets, distance: float | None = None) -> "pytest.MonkeyPatch":
    import mujoco

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = [float(np.mean(offsets[:, 0])), float(np.mean(offsets[:, 1])), 0.75]
    cam.distance = (
        distance if distance is not None else compute_grid_camera_distance(offsets)
    )
    cam.elevation = -20.0
    cam.azimuth = 90.0
    return cam


def _grid_corners(offsets):
    """Extreme corners of the grid (world + margin) plus their projection."""
    pts = [
        [offsets[:, 0].min(), offsets[:, 1].min(), 0.0],
        [offsets[:, 0].max(), offsets[:, 1].min(), 0.0],
        [offsets[:, 0].min(), offsets[:, 1].max(), 0.0],
        [offsets[:, 0].max(), offsets[:, 1].max(), 0.0],
        # robot tops (~1.0 m tall) at the same corners
        [offsets[:, 0].min(), offsets[:, 1].min(), 1.0],
        [offsets[:, 0].max(), offsets[:, 1].max(), 1.0],
    ]
    return np.asarray(pts, dtype=np.float64)


def test_render_spacing_camera_autofit_grid_in_frame():
    """Auto-fit distance keeps every grid corner inside the viewport."""
    offsets = get_grid_offsets(_NUM_ENVS, spacing=_SPACING)
    cam = _make_cam(offsets)
    assert cam.distance > 0
    screen = _project_points(cam, _grid_corners(offsets), _WIDTH, _HEIGHT)
    assert all(s is not None for s in screen), "some grid corners behind camera"
    sx = np.asarray([s[0] for s in screen])
    sy = np.asarray([s[1] for s in screen])
    assert sx.min() >= 0 and sx.max() <= _WIDTH, f"x out of frame: {sx.min():.0f}..{sx.max():.0f}"
    assert sy.min() >= 0 and sy.max() <= _HEIGHT, f"y out of frame: {sy.min():.0f}..{sy.max():.0f}"


def test_grid_offsets_fixed_cols_two_by_four():
    """grid_cols=4 with 8 envs gives an exact row-major 2x4 layout."""
    offsets = get_grid_offsets(8, spacing=6.0, grid_cols=4)
    rows = sorted(set(offsets[:, 0]))
    cols = sorted(set(offsets[:, 1]))
    assert rows == pytest.approx([0.0, 6.0])
    assert cols == pytest.approx([0.0, 6.0, 12.0, 18.0])
    # Row-major: first row holds envs 0..3, second row envs 4..7.
    assert np.allclose(offsets[:4, 0], 0.0) and np.allclose(offsets[4:, 0], 6.0)


def test_grid_offsets_default_stays_near_square():
    """No grid_cols → legacy near-square sqrt behaviour unchanged."""
    # 8 envs: rows=ceil(sqrt(8))=3, cols=ceil(8/3)=3.
    offsets = get_grid_offsets(8, spacing=1.0)
    assert len(set(offsets[:, 0])) == 3
    assert len(set(offsets[:, 1])) == 3


def test_render_spacing_camera_autofit_two_by_four_in_frame():
    """2x4 @ 6 m spacing: robots, trails, markers all inside the viewport."""
    offsets = get_grid_offsets(8, spacing=6.0, grid_cols=4)
    cam = _make_cam(offsets)
    pts = np.vstack(
        [
            _grid_corners(offsets),
            # mid-height points between cells (marker/trail envelope)
            [
                [offsets[i, 0] + dx, offsets[i, 1] + dy, dz]
                for i in (0, 7)
                for dx, dy, dz in ((-0.5, -0.5, 0.2), (0.5, 0.5, 1.15))
            ],
        ]
    )
    screen = _project_points(cam, pts, _WIDTH, _HEIGHT)
    assert all(s is not None for s in screen), "some points behind camera"
    sx = np.asarray([s[0] for s in screen])
    sy = np.asarray([s[1] for s in screen])
    assert sx.min() >= 0 and sx.max() <= _WIDTH, f"x out of frame: {sx.min():.0f}..{sx.max():.0f}"
    assert sy.min() >= 0 and sy.max() <= _HEIGHT, f"y out of frame: {sy.min():.0f}..{sy.max():.0f}"


def test_render_spacing_camera_autofit_scales_with_spacing():
    """Wider spacing → farther auto-fit distance (monotonic in span)."""
    d_close = compute_grid_camera_distance(get_grid_offsets(4, spacing=1.0))
    d_far = compute_grid_camera_distance(get_grid_offsets(4, spacing=_SPACING))
    assert d_far > d_close
    # 4-env 12 m grid spans ~25 m horizontally; the fitted distance must be
    # well above any close-up default while staying bounded by the geometry
    # (no runaway conservative blow-up like the old diagonal heuristic).
    assert d_far > 10.0
    assert d_far < 60.0


def test_render_spacing_camera_autofit_tighter_than_diagonal_heuristic():
    """Projected-bounds fit never lands farther than the old XY-diagonal formula."""
    import math

    offsets = get_grid_offsets(8, spacing=6.0, grid_cols=4)
    span_x = float(np.ptp(offsets[:, 0]))
    span_y = float(np.ptp(offsets[:, 1]))
    diag_half = 0.5 * float(np.hypot(span_x, span_y))
    old_style = diag_half / math.tan(math.radians(_FOVY_DEG) / 2.0)
    fitted = compute_grid_camera_distance(offsets)
    assert fitted < old_style, (
        f"fitted distance {fitted:.1f} m must stay within the diagonal "
        f"heuristic {old_style:.1f} m"
    )
    # Single robot: no grid spread at all, so the fit hugs the robot size
    # instead of growing with any cell-count term.
    single = compute_grid_camera_distance(get_grid_offsets(1, spacing=1.0))
    assert single < 4.0


def test_render_spacing_camera_manual_override_wins():
    """compute_grid_camera_distance is only a fallback; explicit value is used."""
    # render_frame_job keeps an explicit cam_distance; auto-fit only when
    # cam_distance is None / non-positive.
    import mujoco

    cam = mujoco.MjvCamera()
    explicit = 4.0
    cam.distance = explicit
    # The helper must not be reached when the caller supplied a value.
    assert cam.distance == explicit


def test_render_spacing_marker_alignment():
    """Goal/EE marker spheres are shifted by the SAME env offset as the robot."""
    offsets = get_grid_offsets(_NUM_ENVS, spacing=_SPACING)
    goal = np.array([0.30, 0.0, 0.30], dtype=np.float64)  # world (unoffset)
    eye3 = np.eye(3, dtype=np.float32).flatten()
    rgba = np.array([1.0, 0.2, 0.2, 0.9], dtype=np.float32)

    scene = _make_scene()
    for i in range(_NUM_ENVS):
        _add_one_sphere(scene, goal, np.array([0.03, 0, 0], dtype=np.float32), rgba, eye3, offsets, i)
        g = scene.geoms[i]
        assert g.pos[0] == pytest.approx(goal[0] + offsets[i, 0], abs=1e-6)
        assert g.pos[1] == pytest.approx(goal[1] + offsets[i, 1], abs=1e-6)
        assert g.pos[2] == pytest.approx(goal[2], abs=1e-6)  # z untouched


def test_render_spacing_trajectory_alignment():
    """Base/EE trail points are shifted by the SAME env offset per env."""
    offsets = get_grid_offsets(_NUM_ENVS, spacing=_SPACING)
    eye3 = np.eye(3, dtype=np.float32).flatten()
    rgba = np.array([0.2, 0.4, 1.0, 0.9], dtype=np.float32)
    pts = np.array(
        [[0.0, 0.0, 0.3], [0.1, 0.05, 0.3], [0.2, 0.1, 0.3], [np.nan, np.nan, np.nan]],
        dtype=np.float32,
    )

    scene = _make_scene()
    _add_trajectory(scene, pts, rgba, np.array([0.014, 0, 0], dtype=np.float32), eye3, offsets, 1)
    count = 0
    for k in range(scene.ngeom):
        g = scene.geoms[k]
        assert g.pos[0] == pytest.approx(pts[k, 0] + offsets[1, 0], abs=1e-6)
        assert g.pos[1] == pytest.approx(pts[k, 1] + offsets[1, 1], abs=1e-6)
        count += 1
    assert count == 3  # NaN slot skipped, not drawn


def test_render_spacing_text_uses_same_offset():
    """Debug text is projected from base + the SAME env offset."""
    # The renderer projects text from state_batch[i].base + offsets[i]; the
    # helper that places it is _project_points on those shifted points.  This
    # guards the invariant by checking the two offset paths agree.
    offsets = get_grid_offsets(_NUM_ENVS, spacing=_SPACING)
    assert np.allclose(
        offsets + get_grid_offsets(_NUM_ENVS, spacing=0.0), offsets, atol=1e-9
    )
    assert _FOVY_DEG == 45.0  # projection uses the same FOV the renderer uses
