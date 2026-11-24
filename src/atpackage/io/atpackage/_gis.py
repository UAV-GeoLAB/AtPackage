"""GeoPackage layers derived from the package data for display in GIS software.

GIS software can't decode the float64 blobs of the data tables, so these
layers repeat the essentials as plain columns plus a Point Z geometry. They
are rebuilt from the data tables on every write.

Coordinates are in the GIS CRS of the block (see :mod:`atpackage.crs`): for a
geocentric or topocentric block, ``position_x/y/z`` are WGS 84 longitude,
latitude and ellipsoidal height, and omega-phi-kappa refer to the local
east-north-up frame at each image; otherwise they are the block's own
coordinates and axes.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from ...crs import (
    gis_crs,
    is_geocentric_or_topocentric,
    to_gis_coordinates,
    world_to_ecef,
)
from ...rotation import camera_rotvec_to_opk, ecef_to_enu
from ...schema.geopackage import SpatialRefSys, point_z_geometries, update_contents
from ...schema.tables import FLOAT64_BLOB_DTYPE, GIS_GCPS, GIS_IMAGES, VEC3, Table


def _blobs_to_array(blobs: pd.Series, length: int) -> np.ndarray:
    """Stack non-null float64 blobs into an array of shape (N, length)."""
    data = np.frombuffer(b"".join(blobs), dtype=FLOAT64_BLOB_DTYPE)
    return data.reshape(-1, length)


def _xyz_columns(prefix: str, xyz: np.ndarray) -> dict[str, np.ndarray]:
    return {f"{prefix}_{axis}": xyz[:, i] for i, axis in enumerate("xyz")}


def _with_geometry(df: pd.DataFrame, xyz: np.ndarray, srs_id: int) -> pd.DataFrame:
    df.insert(0, "geom", pd.Series(point_z_geometries(xyz, srs_id), dtype=object))
    return df


def gis_images_frame(images: pd.DataFrame, crs: SpatialRefSys) -> pd.DataFrame:
    """Build the GIS_IMAGES layer from a validated IMAGES DataFrame."""
    position = to_gis_coordinates(_blobs_to_array(images["position"], VEC3), crs)
    rotvec = _blobs_to_array(images["rotation"], VEC3)
    world_to_local = None
    if is_geocentric_or_topocentric(crs) and len(position):
        ecef_to_local = ecef_to_enu(position[:, 0], position[:, 1], degrees=True)
        world_to_local = ecef_to_local * world_to_ecef(crs)
    opk = camera_rotvec_to_opk(
        rotvec, degrees=True, world_to_local=world_to_local
    ).reshape(-1, 3)

    attributes = images[["name", "path", "camera_index", "timestamp", "strip_index"]]
    df = attributes.reset_index(drop=True).assign(
        **_xyz_columns("position", position),
        rotation_omega=opk[:, 0],
        rotation_phi=opk[:, 1],
        rotation_kappa=opk[:, 2],
    )
    srs_id = gis_crs(crs).srs_id
    return GIS_IMAGES.schema.validate(_with_geometry(df, position, srs_id))


def gis_gcps_frame(gcps: pd.DataFrame, crs: SpatialRefSys) -> pd.DataFrame:
    """Build the GIS_GCPS layer from a validated GCPS DataFrame."""
    position = to_gis_coordinates(_blobs_to_array(gcps["position"], VEC3), crs)
    prior = to_gis_coordinates(_blobs_to_array(gcps["position_prior"], VEC3), crs)
    df = gcps[["name", "type"]].reset_index(drop=True).assign(
        **_xyz_columns("position", position),
        **_xyz_columns("position_prior", prior),
    )
    srs_id = gis_crs(crs).srs_id
    return GIS_GCPS.schema.validate(_with_geometry(df, position, srs_id))


def _replace_layer(con: sqlite3.Connection, table: Table, df: pd.DataFrame) -> None:
    extent = None
    if not df.empty:
        extent = (
            df["position_x"].min(),
            df["position_y"].min(),
            df["position_x"].max(),
            df["position_y"].max(),
        )
    with con:
        con.execute(f"DELETE FROM {table.name}")
    table.write(con, df)
    with con:
        update_contents(con, table.name, extent)


def write_gis_images(
    con: sqlite3.Connection, images: pd.DataFrame, crs: SpatialRefSys
) -> None:
    """Replace the GIS_IMAGES layer with one built from ``images``."""
    _replace_layer(con, GIS_IMAGES, gis_images_frame(images, crs))


def write_gis_gcps(
    con: sqlite3.Connection, gcps: pd.DataFrame, crs: SpatialRefSys
) -> None:
    """Replace the GIS_GCPS layer with one built from ``gcps``."""
    _replace_layer(con, GIS_GCPS, gis_gcps_frame(gcps, crs))
