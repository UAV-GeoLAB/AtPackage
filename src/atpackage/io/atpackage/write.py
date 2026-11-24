from __future__ import annotations

import os
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path

import pandas as pd

from ...schema.geopackage import UNDEFINED_CARTESIAN, SpatialRefSys
from ...schema.tables import (
    GCPS,
    IMAGES,
    METADATA,
    TABLES,
    create_all,
    validate_user_metadata,
)
from ._gis import write_gis_gcps, write_gis_images

FILE_EXTENSION = ".at.gpkg"


def write_atpackage(
    path: str | os.PathLike[str],
    tables: Mapping[str, pd.DataFrame],
    crs: SpatialRefSys = UNDEFINED_CARTESIAN,
    *,
    overwrite: bool = False,
) -> None:
    """Write the package tables to a new ``.at.gpkg`` file.

    ``tables`` maps table names to DataFrames; missing tables are written
    empty. ``crs`` is the block CRS of world X, Y, Z. The metadata table may
    only hold user keys; the reserved keys (format version, CRS) are added
    here. The file is built next to ``path``
    and moved into place only when complete, so a failed write never leaves a
    half-written package behind.
    """
    path = Path(path)
    if not path.name.endswith(FILE_EXTENSION):
        raise ValueError(f"{path}: atpackage files must end with {FILE_EXTENSION}")
    unknown = set(tables) - {table.name for table in TABLES}
    if unknown:
        raise ValueError(f"Unknown tables: {sorted(unknown)}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")

    frames = {
        table.name: tables.get(table.name, table.empty()) for table in TABLES
    }
    validate_user_metadata(frames[METADATA.name])
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        with closing(sqlite3.connect(tmp)) as con:
            create_all(con, crs)
            for table in TABLES:
                table.write(con, frames[table.name])
            write_gis_images(con, frames[IMAGES.name], crs)
            write_gis_gcps(con, frames[GCPS.name], crs)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
