"""Coordinate reference system of a block and of its GIS layers.

The data tables hold world coordinates in a single block CRS, which is one of:

* a compound CRS: horizontal + vertical, e.g. ``"EPSG:2180+9651"``;
* a 3D CRS: geocentric (ECEF, e.g. ``"EPSG:4978"``), geographic 3D
  (``"EPSG:4979"``) or topocentric (a local east-north-up frame at a given
  origin, WKT2 only);
* a 2D CRS, when the height reference of world Z is unknown;
* the undefined Cartesian CRS of GeoPackage, for unreferenced blocks.

The file stores it in ``gpkg_spatial_ref_sys`` as WKT2, plus WKT1 where WKT1
can express it (see :mod:`atpackage.schema.geopackage`).

GIS software can't display geocentric or topocentric coordinates, so the GIS
layers of such a block are in WGS 84 (longitude, latitude, ellipsoidal height)
and their omega-phi-kappa refer to the local east-north-up frame at each point.
The GIS layers of a compound block use its horizontal CRS; other blocks keep
their CRS in the GIS layers.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from pyproj import CRS, Transformer
from pyproj.enums import WktVersion
from pyproj.exceptions import CRSError
from scipy.spatial.transform import Rotation

from .schema.geopackage import UNDEFINED, WGS84, SpatialRefSys

# Anything pyproj understands ("EPSG:2180+9651", 2180, WKT, pyproj.CRS, ...).
type CrsLike = SpatialRefSys | CRS | str | int

# srs_id of a CRS without a numeric authority code (the first one GDAL uses).
CUSTOM_SRS_ID = 100000

# EPSG methods of topocentric conversions: geocentric/topocentric (9836) and
# geographic/topocentric (9837).
_TOPOCENTRIC_METHOD_CODES = frozenset({"9836", "9837"})

# 3D WGS 84 (lon, lat, ellipsoidal height) and geocentric WGS 84.
_WGS84_3D = CRS.from_epsg(4979)
_WGS84_GEOCENTRIC = CRS.from_epsg(4978)


def _numeric_authority(crs: CRS) -> tuple[str, int] | None:
    authority = crs.to_authority()
    if authority is None or not authority[1].isdigit():
        return None
    return authority[0], int(authority[1])


def _wkt(crs: CRS, version: WktVersion) -> str:
    try:
        return crs.to_wkt(version) or UNDEFINED
    except CRSError:  # the CRS can't be expressed in this WKT version
        return UNDEFINED


def as_spatial_ref_sys(crs: CrsLike, srs_id: int | None = None) -> SpatialRefSys:
    """GeoPackage CRS definition from any pyproj CRS input.

    ``srs_id`` defaults to the CRS's numeric authority code (e.g. the EPSG
    code), or :data:`CUSTOM_SRS_ID` for CRSs without one (compound CRSs
    built from two codes, topocentric CRSs, custom projections). The
    definition is stored as WKT2, and also as WKT1 when WKT1 can express it.
    """
    if isinstance(crs, SpatialRefSys):
        return crs
    crs = CRS.from_user_input(crs)
    authority = _numeric_authority(crs)
    if srs_id is None:
        srs_id = authority[1] if authority is not None else CUSTOM_SRS_ID
    organization, coordsys_id = authority if authority is not None else ("NONE", srs_id)
    return SpatialRefSys(
        crs.name,
        srs_id,
        organization,
        coordsys_id,
        definition=_wkt(crs, WktVersion.WKT1_GDAL),
        definition_12_063=_wkt(crs, WktVersion.WKT2_2019),
    )


def to_pyproj(srs: SpatialRefSys) -> CRS | None:
    """pyproj CRS of a GeoPackage definition; ``None`` for the undefined ones.

    Tries the authority code in the PROJ database, then WKT2, then WKT1.
    """
    if srs.organization.upper() == "EPSG":
        try:
            return CRS.from_epsg(srs.organization_coordsys_id)
        except CRSError:
            pass  # unknown to this PROJ database; fall back to the WKT
    wkts = [
        wkt
        for wkt in (srs.definition_12_063, srs.definition)
        if wkt.strip().lower() != UNDEFINED
    ]
    error: CRSError | None = None
    for wkt in wkts:
        try:
            return CRS.from_wkt(wkt)
        except CRSError as wkt_error:
            error = error or wkt_error
    if error is not None:
        message = f"Can't interpret the definition of {srs.srs_name!r}"
        raise ValueError(message) from error
    return None


def _is_topocentric(crs: CRS) -> bool:
    operation = crs.coordinate_operation
    if operation is None:
        return False
    if operation.method_code in _TOPOCENTRIC_METHOD_CODES:
        return True
    return "topocentric" in operation.method_name.lower()


def is_geocentric_or_topocentric(srs: SpatialRefSys) -> bool:
    """``srs`` is an earth-fixed 3D Cartesian CRS that GIS software can't show."""
    crs = to_pyproj(srs)
    return crs is not None and (crs.is_geocentric or _is_topocentric(crs))


def check_block_crs(srs: SpatialRefSys) -> None:
    """A block CRS must define world X and Y: a vertical CRS alone can't."""
    crs = to_pyproj(srs)
    if crs is not None and len(crs.axis_info) < 2:
        raise ValueError(
            f"{srs.srs_name!r} has {len(crs.axis_info)} axis; the block CRS "
            "needs a horizontal part, e.g. a compound CRS such as "
            '"EPSG:2180+9651"'
        )


def gis_crs(srs: SpatialRefSys) -> SpatialRefSys:
    """CRS of the GIS layers of a block whose world coordinates are in ``srs``."""
    if is_geocentric_or_topocentric(srs):
        return WGS84
    crs = to_pyproj(srs)
    if crs is None or not crs.is_compound:
        return srs
    horizontal = crs.sub_crs_list[0]
    authority = _numeric_authority(horizontal)
    if authority is not None:  # canonical definition from the PROJ database
        return as_spatial_ref_sys(f"{authority[0]}:{authority[1]}")
    # A horizontal CRS without a code must not take the block's srs_id.
    return as_spatial_ref_sys(horizontal, srs.srs_id + 1)


def to_gis_coordinates(xyz: ArrayLike, srs: SpatialRefSys) -> np.ndarray:
    """Block world coordinates (N, 3) -> GIS layer coordinates (N, 3).

    Geocentric and topocentric blocks become WGS 84 longitude, latitude
    (degrees) and ellipsoidal height; other blocks are returned unchanged.
    """
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if not is_geocentric_or_topocentric(srs):
        return xyz
    transformer = Transformer.from_crs(to_pyproj(srs), _WGS84_3D, always_xy=True)
    return np.column_stack(transformer.transform(xyz[:, 0], xyz[:, 1], xyz[:, 2]))


def world_to_ecef(srs: SpatialRefSys) -> Rotation:
    """Rotation from the axes of a geocentric or topocentric CRS to ECEF axes.

    Both are rigid transformations of ECEF, so the rotation is the same at
    every point; it is identity for geocentric CRSs.
    """
    crs = to_pyproj(srs)
    if crs is None or not (crs.is_geocentric or _is_topocentric(crs)):
        raise ValueError(f"{srs.srs_name!r} is neither geocentric nor topocentric")
    if crs.is_geocentric:
        return Rotation.identity()
    transformer = Transformer.from_crs(crs, _WGS84_GEOCENTRIC, always_xy=True)
    step = 1000.0  # metres; the transformation is affine, so any step works
    points = np.vstack([np.zeros(3), step * np.eye(3)])
    ecef = np.column_stack(transformer.transform(*points.T))
    matrix = (ecef[1:] - ecef[0]).T / step  # columns: world axes in ECEF
    if np.linalg.det(matrix) <= 0:
        raise ValueError(f"{srs.srs_name!r} has left-handed axes")
    return Rotation.from_matrix(matrix)
