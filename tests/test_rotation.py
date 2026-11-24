import numpy as np
import pytest

from atpackage.rotation import camera_rotvec_to_opk, opk_to_camera_rotvec

PHOTO_TO_CAMERA = np.diag([1.0, -1.0, -1.0])


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotvec_to_matrix(rotvec) -> np.ndarray:
    """Rodrigues formula, independent of scipy, to pin the convention."""
    r = np.asarray(rotvec, dtype=float)
    theta = np.linalg.norm(r)
    if theta == 0:
        return np.eye(3)
    k = r / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * kx @ kx


def test_nadir_north_up_camera_is_zero_opk():
    # Camera looks down (z -> world -Z), image top points north (y down -> -Y).
    np.testing.assert_allclose(camera_rotvec_to_opk([np.pi, 0, 0]), 0, atol=1e-15)


def test_nadir_south_up_camera_has_kappa_180():
    opk = camera_rotvec_to_opk([0, np.pi, 0], degrees=True)
    np.testing.assert_allclose(opk[:2], 0, atol=1e-12)
    np.testing.assert_allclose(abs(opk[2]), 180, atol=1e-12)


@pytest.mark.parametrize(
    "opk", [(0.1, 0.2, 0.3), (-2.5, 1.2, 3.0), (0.0, -0.4, -1.7)]
)
def test_opk_matches_photogrammetric_definition(opk):
    # Camera -> world matrix built by hand: R_photo_to_world @ diag(1, -1, -1).
    o, p, k = opk
    camera_to_world = rot_x(o) @ rot_y(p) @ rot_z(k) @ PHOTO_TO_CAMERA
    rotvec = opk_to_camera_rotvec(opk)
    np.testing.assert_allclose(rotvec_to_matrix(rotvec), camera_to_world, atol=1e-14)
    np.testing.assert_allclose(camera_rotvec_to_opk(rotvec), opk, atol=1e-13)


def test_round_trip_is_vectorized_and_canonical():
    rng = np.random.default_rng(1)
    opk = np.column_stack([
        rng.uniform(-180, 180, 1000),
        rng.uniform(-89.9, 89.9, 1000),
        rng.uniform(-180, 180, 1000),
    ])
    rotvec = opk_to_camera_rotvec(opk, degrees=True)
    assert rotvec.shape == (1000, 3)
    assert np.all(np.linalg.norm(rotvec, axis=1) <= np.pi + 1e-12)
    np.testing.assert_allclose(
        camera_rotvec_to_opk(rotvec, degrees=True), opk, atol=1e-9
    )


def test_empty_input():
    assert camera_rotvec_to_opk(np.zeros((0, 3))).shape == (0, 3)
