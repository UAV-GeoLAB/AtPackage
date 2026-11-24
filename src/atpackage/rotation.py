"""Rotation conversions between the package convention and omega-phi-kappa.

Package convention (see :mod:`atpackage.schema.tables`): image rotations are
rotation vectors mapping camera -> world, with camera axes x right, y down,
z along the viewing direction.

Omega-phi-kappa follow the classic photogrammetric definition: the rotation
from the photogrammetric image frame (x right, y up, z pointing back out of the
camera) to world is ``R = Rx(omega) @ Ry(phi) @ Rz(kappa)``, i.e. intrinsic
``"XYZ"`` Euler angles in scipy. A nadir image with its top edge pointing to
world +Y has omega = phi = kappa = 0.

Functions accept a single value of shape ``(3,)`` or an array of shape ``(N, 3)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

# scipy Euler sequence for omega-phi-kappa: intrinsic X, then Y, then Z.
OPK_SEQUENCE = "XYZ"

# Camera frame (x right, y down, z forward) -> photogrammetric image frame
# (x right, y up, z backward): diag(1, -1, -1), a half turn about x.
PHOTO_TO_CAMERA = Rotation.from_rotvec([np.pi, 0.0, 0.0])


def camera_rotvec_to_opk(
    rotvec: ArrayLike,
    *,
    degrees: bool = False,
    world_to_local: Rotation | None = None,
) -> np.ndarray:
    """Omega, phi, kappa of camera -> world rotation vectors.

    With ``world_to_local`` (e.g. :func:`ecef_to_enu`), the angles refer to
    that local frame instead of the world axes.

    Omega and kappa are in [-180, 180], phi in [-90, 90] (degrees, or the
    radian equivalents). At phi = +-90 (gimbal lock) scipy warns and sets
    kappa to 0.
    """
    camera_to_frame = Rotation.from_rotvec(rotvec)
    if world_to_local is not None:
        camera_to_frame = world_to_local * camera_to_frame
    photo_to_frame = camera_to_frame * PHOTO_TO_CAMERA
    return photo_to_frame.as_euler(OPK_SEQUENCE, degrees=degrees)


def ecef_to_enu(lon: ArrayLike, lat: ArrayLike, *, degrees: bool = False) -> Rotation:
    """Rotations from geocentric (ECEF) axes to local east-north-up axes.

    ``lon`` and ``lat`` are geodetic longitude and latitude of the points.
    """
    lon = np.radians(lon) if degrees else np.asarray(lon, dtype=np.float64)
    lat = np.radians(lat) if degrees else np.asarray(lat, dtype=np.float64)
    sl, cl = np.sin(lon), np.cos(lon)
    sp, cp = np.sin(lat), np.cos(lat)
    zero = np.zeros_like(sl)
    # Rows: east, north and up unit vectors expressed in ECEF.
    matrix = np.stack([
        np.stack([-sl, cl, zero], axis=-1),
        np.stack([-sp * cl, -sp * sl, cp], axis=-1),
        np.stack([cp * cl, cp * sl, sp], axis=-1),
    ], axis=-2)
    return Rotation.from_matrix(matrix)


def opk_to_camera_rotvec(opk: ArrayLike, *, degrees: bool = False) -> np.ndarray:
    """Camera -> world rotation vectors (angle in [0, pi]) from omega-phi-kappa."""
    photo_to_world = Rotation.from_euler(OPK_SEQUENCE, opk, degrees=degrees)
    camera_to_world = photo_to_world * PHOTO_TO_CAMERA.inv()
    return camera_to_world.as_rotvec()
