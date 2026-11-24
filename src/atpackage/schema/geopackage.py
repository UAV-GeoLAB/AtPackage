"""Minimal OGC GeoPackage 1.4 support.

Makes the package file a valid GeoPackage, so GIS software (QGIS, GDAL/OGR,
ArcGIS) opens it and shows the tables registered in ``gpkg_contents``. The
package's own data tables are left unregistered, which GeoPackage allows.

Every file uses the WKT for Coordinate Reference Systems extension
(``gpkg_crs_wkt_1_1``), so CRSs that WKT1 can't express, such as topocentric
ones, are stored as WKT2.
"""

from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

APPLICATION_ID = 0x47504B47  # "GPKG"
USER_VERSION = 10400  # GeoPackage 1.4.0

# GeoPackage DATETIME format: ISO 8601, UTC, millisecond precision.
DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


UNDEFINED = "undefined"


@dataclass(frozen=True)
class SpatialRefSys:
    """One row of ``gpkg_spatial_ref_sys``.

    ``definition`` is WKT1 (OGC 01-009); ``definition_12_063`` is WKT2
    (ISO 19162:2019), added by the ``gpkg_crs_wkt_1_1`` extension. Either is
    ``"undefined"`` when the CRS can't be written in that form; readers prefer
    WKT2. ``epoch`` is the coordinate epoch (decimal year) of a dynamic CRS.
    """

    srs_name: str
    srs_id: int
    organization: str
    organization_coordsys_id: int
    definition: str
    description: str | None = None
    definition_12_063: str = UNDEFINED
    epoch: float | None = None


WGS84 = SpatialRefSys(
    "WGS 84 geodetic",
    4326,
    "EPSG",
    4326,
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,'
    'AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,'
    'AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,'
    'AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],'
    'AUTHORITY["EPSG","4326"]]',
    "longitude/latitude coordinates in decimal degrees on the WGS 84 spheroid",
    definition_12_063=(
        'GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",'
        'ELLIPSOID["WGS 84",6378137,298.257223563,LENGTHUNIT["metre",1]]],'
        'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],'
        "CS[ellipsoidal,2],"
        'AXIS["geodetic latitude (Lat)",north,ORDER[1],'
        'ANGLEUNIT["degree",0.0174532925199433]],'
        'AXIS["geodetic longitude (Lon)",east,ORDER[2],'
        'ANGLEUNIT["degree",0.0174532925199433]],'
        'ID["EPSG",4326]]'
    ),
)
UNDEFINED_CARTESIAN = SpatialRefSys(
    "Undefined cartesian SRS",
    -1,
    "NONE",
    -1,
    UNDEFINED,
    "undefined cartesian coordinate reference system",
)
UNDEFINED_GEOGRAPHIC = SpatialRefSys(
    "Undefined geographic SRS",
    0,
    "NONE",
    0,
    UNDEFINED,
    "undefined geographic coordinate reference system",
)
# Rows every GeoPackage must contain.
REQUIRED_SRS = (WGS84, UNDEFINED_CARTESIAN, UNDEFINED_GEOGRAPHIC)

# WKT for Coordinate Reference Systems extension (OGC 21-057), version 1.1:
# adds the definition_12_063 (WKT2) and epoch columns to gpkg_spatial_ref_sys.
CRS_WKT_EXTENSION = "gpkg_crs_wkt_1_1"
_CRS_WKT_EXTENSION_URL = "http://www.geopackage.org/spec/#extension_crs_wkt"

_SRS_COLUMNS = (
    "srs_name, srs_id, organization, organization_coordsys_id,"
    " definition, description, definition_12_063, epoch"
)

_CORE_TABLES_SQL = (
    """CREATE TABLE gpkg_spatial_ref_sys (
    srs_name TEXT NOT NULL,
    srs_id INTEGER PRIMARY KEY,
    organization TEXT NOT NULL,
    organization_coordsys_id INTEGER NOT NULL,
    definition TEXT NOT NULL,
    description TEXT,
    definition_12_063 TEXT NOT NULL,
    epoch DOUBLE
)""",
    """CREATE TABLE gpkg_extensions (
    table_name TEXT,
    column_name TEXT,
    extension_name TEXT NOT NULL,
    definition TEXT NOT NULL,
    scope TEXT NOT NULL,
    CONSTRAINT ge_tce UNIQUE (table_name, column_name, extension_name)
)""",
    """CREATE TABLE gpkg_contents (
    table_name TEXT NOT NULL PRIMARY KEY,
    data_type TEXT NOT NULL,
    identifier TEXT UNIQUE,
    description TEXT DEFAULT '',
    last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    min_x DOUBLE,
    min_y DOUBLE,
    max_x DOUBLE,
    max_y DOUBLE,
    srs_id INTEGER,
    CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id)
        REFERENCES gpkg_spatial_ref_sys(srs_id)
)""",
    """CREATE TABLE gpkg_geometry_columns (
    table_name TEXT NOT NULL,
    column_name TEXT NOT NULL,
    geometry_type_name TEXT NOT NULL,
    srs_id INTEGER NOT NULL,
    z TINYINT NOT NULL,
    m TINYINT NOT NULL,
    CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),
    CONSTRAINT uk_gc_table_name UNIQUE (table_name),
    CONSTRAINT fk_gc_tn FOREIGN KEY (table_name)
        REFERENCES gpkg_contents(table_name),
    CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id)
        REFERENCES gpkg_spatial_ref_sys(srs_id)
)""",
)


def init_geopackage(con: sqlite3.Connection) -> None:
    """Set the GeoPackage header and create the core tables in an empty file."""
    con.execute(f"PRAGMA application_id = {APPLICATION_ID}")
    con.execute(f"PRAGMA user_version = {USER_VERSION}")
    for sql in _CORE_TABLES_SQL:
        con.execute(sql)
    con.executemany(
        "INSERT INTO gpkg_extensions (table_name, column_name, extension_name,"
        " definition, scope) VALUES ('gpkg_spatial_ref_sys', ?, ?, ?, 'read-write')",
        [
            (column, CRS_WKT_EXTENSION, _CRS_WKT_EXTENSION_URL)
            for column in ("definition_12_063", "epoch")
        ],
    )
    for srs in REQUIRED_SRS:
        add_spatial_ref_sys(con, srs)


def read_spatial_ref_sys(con: sqlite3.Connection, srs_id: int) -> SpatialRefSys | None:
    row = con.execute(
        f"SELECT {_SRS_COLUMNS} FROM gpkg_spatial_ref_sys WHERE srs_id = ?",
        (srs_id,),
    ).fetchone()
    return None if row is None else SpatialRefSys(*row)


def _same_system(a: SpatialRefSys, b: SpatialRefSys) -> bool:
    if a.organization.upper() != b.organization.upper():
        return False
    if a.organization.upper() != "NONE":
        return a.organization_coordsys_id == b.organization_coordsys_id
    # No authority to compare: the definitions must match.
    return (a.definition, a.definition_12_063) == (b.definition, b.definition_12_063)


def add_spatial_ref_sys(con: sqlite3.Connection, srs: SpatialRefSys) -> None:
    """Insert ``srs``; keep an existing row with the same ``srs_id`` and system.

    Rows are the same system if they share the authority code, or, for
    systems without one (organization ``NONE``), the definitions. Raises
    ``ValueError`` if ``srs_id`` is taken by a different system.
    """
    existing = read_spatial_ref_sys(con, srs.srs_id)
    if existing is not None:
        if not _same_system(existing, srs):
            raise ValueError(
                f"srs_id {srs.srs_id} is already used by {existing.srs_name!r} "
                f"({existing.organization}:{existing.organization_coordsys_id})"
            )
        return
    con.execute(
        f"INSERT INTO gpkg_spatial_ref_sys ({_SRS_COLUMNS})"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            srs.srs_name,
            srs.srs_id,
            srs.organization,
            srs.organization_coordsys_id,
            srs.definition,
            srs.description,
            srs.definition_12_063,
            srs.epoch,
        ),
    )


def register_features(
    con: sqlite3.Connection,
    table_name: str,
    column_name: str,
    geometry_type_name: str,
    srs_id: int,
    *,
    z: int,
    m: int = 0,
    identifier: str | None = None,
    description: str = "",
) -> None:
    """Register a feature table in ``gpkg_contents`` and ``gpkg_geometry_columns``.

    ``z`` and ``m``: 0 = prohibited, 1 = mandatory, 2 = optional.
    """
    con.execute(
        "INSERT INTO gpkg_contents (table_name, data_type, identifier, description,"
        " srs_id) VALUES (?, 'features', ?, ?, ?)",
        (table_name, identifier or table_name, description, srs_id),
    )
    con.execute(
        "INSERT INTO gpkg_geometry_columns (table_name, column_name,"
        " geometry_type_name, srs_id, z, m) VALUES (?, ?, ?, ?, ?, ?)",
        (table_name, column_name, geometry_type_name, srs_id, z, m),
    )


def feature_srs_id(con: sqlite3.Connection, table_name: str) -> int:
    row = con.execute(
        "SELECT srs_id FROM gpkg_geometry_columns WHERE table_name = ?",
        (table_name,),
    ).fetchone()
    if row is None:
        raise LookupError(f"{table_name} is not a registered feature table")
    return int(row[0])


def update_contents(
    con: sqlite3.Connection,
    table_name: str,
    extent: tuple[float, float, float, float] | None,
) -> None:
    """Refresh ``last_change`` and the (min_x, min_y, max_x, max_y) extent."""
    min_x, min_y, max_x, max_y = extent if extent is not None else (None,) * 4
    con.execute(
        "UPDATE gpkg_contents SET"
        " last_change = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),"
        " min_x = ?, min_y = ?, max_x = ?, max_y = ?"
        " WHERE table_name = ?",
        (min_x, min_y, max_x, max_y, table_name),
    )


# GeoPackage binary header: magic "GP", version 0 (= 1.0), flags, srs_id.
# Flags 0b00000001: little-endian header, no envelope, not empty, standard type.
_GPKG_HEADER = struct.Struct("<2sBBi")
_GPKG_FLAGS = 0b0000_0001
# ISO WKB header: little-endian byte order, geometry type 1001 = Point Z.
_WKB_POINT_Z = struct.pack("<BI", 1, 1001)
WKB_POINT_Z_SIZE = _GPKG_HEADER.size + len(_WKB_POINT_Z) + 3 * 8


def point_z_geometries(xyz: ArrayLike, srs_id: int) -> list[bytes]:
    """GeoPackage binary Point Z geometries from an array of shape (N, 3)."""
    header = _GPKG_HEADER.pack(b"GP", 0, _GPKG_FLAGS, srs_id) + _WKB_POINT_Z
    coords = np.asarray(xyz, dtype="<f8").reshape(-1, 3)
    return [header + row.tobytes() for row in coords]


def is_point_z_geometry(value: object) -> bool:
    """``value`` is a non-empty GeoPackage binary Point Z without envelope."""
    return (
        isinstance(value, (bytes, bytearray))
        and len(value) == WKB_POINT_Z_SIZE
        and value[:2] == b"GP"
        and value[3] == _GPKG_FLAGS
        and value[_GPKG_HEADER.size : _GPKG_HEADER.size + 5] == _WKB_POINT_Z
    )
