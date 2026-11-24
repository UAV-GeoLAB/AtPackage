from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from ...schema.geopackage import APPLICATION_ID, SpatialRefSys, read_spatial_ref_sys
from ...schema.tables import (
    FORMAT_VERSION,
    GIS_IMAGES,
    METADATA,
    RESERVED_METADATA_PREFIX,
    TABLES,
    MetadataKey,
)
from .._package_data import PackageData


def read_atpackage(path: str | os.PathLike[str]) -> PackageData:
    """Read the package tables and CRSs from an atpackage file.

    The GIS layers are derived data and are not read back. Reserved metadata
    keys are interpreted and removed from the returned metadata table.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    uri = path.resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as con:
        _check_atpackage(con, path)
        tables = {table.name: table.read(con) for table in TABLES}
        user_metadata, reserved = _split_metadata(tables[METADATA.name])
        tables[METADATA.name] = user_metadata
        _check_version(reserved, path)
        crs = _crs(con, reserved, path)
    return PackageData(tables, crs)


def _check_atpackage(con: sqlite3.Connection, path: Path) -> None:
    try:
        (application_id,) = con.execute("PRAGMA application_id").fetchone()
    except sqlite3.DatabaseError as error:
        raise ValueError(f"{path} is not an atpackage file: {error}") from error
    if application_id != APPLICATION_ID:
        raise ValueError(f"{path} is not an atpackage file (not a GeoPackage)")
    present = {row[0] for row in con.execute("SELECT name FROM sqlite_master")}
    missing = [t.name for t in (*TABLES, GIS_IMAGES) if t.name not in present]
    if missing:
        raise ValueError(f"{path} is not an atpackage file (missing {missing})")


def _split_metadata(metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    is_reserved = metadata["key"].str.startswith(RESERVED_METADATA_PREFIX)
    reserved = metadata[is_reserved]
    user = metadata[~is_reserved].reset_index(drop=True)
    return user, dict(zip(reserved["key"], reserved["value"], strict=True))


def _check_version(reserved: dict[str, str], path: Path) -> None:
    version = reserved.get(MetadataKey.FORMAT_VERSION)
    if version is None:
        raise ValueError(f"{path}: no {MetadataKey.FORMAT_VERSION} in metadata")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"{path}: format version {version} is not supported "
            f"(this atpackage reads {FORMAT_VERSION})"
        )


def _crs(
    con: sqlite3.Connection, reserved: dict[str, str], path: Path
) -> SpatialRefSys:
    key = MetadataKey.SRS_ID
    if key not in reserved:
        raise ValueError(f"{path}: no {key} in metadata")
    srs = read_spatial_ref_sys(con, int(reserved[key]))
    if srs is None:
        raise ValueError(
            f"{path}: {key} = {reserved[key]} is not in gpkg_spatial_ref_sys"
        )
    return srs
