"""Table definitions for the atpackage SQLite format.

Every table is described once, as a sequence of :class:`Col` specs. From that
single description we derive:

* the ``CREATE TABLE`` statement (with ``NOT NULL`` and ``CHECK`` constraints,
  so the file stays valid even when edited outside this package),
* the pandera schema used to validate DataFrames,
* the dtype map used when reading the table into pandas.

The file is an OGC GeoPackage (see :mod:`.geopackage`). The package's data
tables are ``STRICT`` SQLite tables that GeoPackage leaves unregistered; the
GIS tables are GeoPackage feature tables, registered in ``gpkg_contents``.

Integer columns always use pandas nullable extension dtypes (``UInt8``,
``Int64``, ...), so SQL ``NULL`` arrives as ``<NA>`` instead of silently turning
the column into ``float64``. SQLite ``INTEGER`` is signed 64-bit, so unsigned
64-bit columns are not supported; use ``Int64`` with ``min_value=0`` instead.

Image pose convention (as in Agisoft Metashape):

* ``position`` is the camera projection centre C in world coordinates.
* ``rotation`` is a rotation vector r (axis * angle, radians, angle in [0, pi])
  whose rotation matrix R = exp([r]x) maps camera to world:
  ``X_world = R @ X_camera + C``.
* Camera axes: x right, y down, z along the viewing direction.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa

from ..crs import check_block_crs, gis_crs
from .enums import (
    CameraDistortionModelType,
    CameraDistortionModelTypeKeys,
    GcpTypeValues,
)
from .geopackage import (
    UNDEFINED_CARTESIAN,
    SpatialRefSys,
    add_spatial_ref_sys,
    init_geopackage,
    is_point_z_geometry,
    register_features,
)

FLOAT64_BLOB_DTYPE = np.dtype("<f8")
_SQLITE_INT_MAX = 2**63 - 1


def is_float64_blob(value: Any, length: int | None = None) -> bool:
    """``value`` is bytes holding a float64 array (of ``length`` items if given)."""
    if not isinstance(value, (bytes, bytearray)):
        return False
    itemsize = FLOAT64_BLOB_DTYPE.itemsize
    if length is None:
        return len(value) % itemsize == 0
    return len(value) == length * itemsize


class SqlType(StrEnum):
    """Column types; all are valid GeoPackage data types."""

    INTEGER = "INTEGER"
    SMALLINT = "SMALLINT"  # GeoPackage 16-bit signed integer
    REAL = "REAL"
    TEXT = "TEXT"
    BLOB = "BLOB"
    DATETIME = "DATETIME"  # GeoPackage ISO 8601 UTC text
    POINT = "POINT"  # GeoPackage geometry


# Types allowed in SQLite STRICT tables.
_STRICT_TYPES = {SqlType.INTEGER, SqlType.REAL, SqlType.TEXT, SqlType.BLOB}


@dataclass(frozen=True)
class Col:
    """One column: its SQL storage, its pandas dtype and its constraints."""

    name: str
    sql_type: SqlType
    dtype: Any  # dtype of the column in a DataFrame (pandas / pandera notation)
    nullable: bool = False
    sql_checks: tuple[str, ...] = ()
    checks: tuple[pa.Check, ...] = ()
    parse_dates: bool = False  # stored as ISO-8601 TEXT, parsed on read
    # Converts the DataFrame column to its stored form on write.
    encode: Callable[[pd.Series], pd.Series] | None = None
    unique: bool = False

    def ddl(self) -> str:
        parts = [self.name, self.sql_type.value]
        if not self.nullable:
            parts.append("NOT NULL")
        if self.unique:
            parts.append("UNIQUE")
        parts += [f"CHECK ({check})" for check in self.sql_checks]
        return " ".join(parts)

    def pandera_column(self) -> pa.Column:
        return pa.Column(
            self.dtype,
            nullable=self.nullable,
            unique=self.unique,
            checks=list(self.checks),
        )


# --- column factories ---------------------------------------------------------


def int_col(
    name: str,
    dtype: str,
    *,
    nullable: bool = False,
    min_value: int | None = None,
    allowed: Iterable[int] | None = None,
    sql_type: SqlType = SqlType.INTEGER,
) -> Col:
    """Integer column backed by a pandas nullable dtype such as ``"UInt8"``.

    The value range implied by ``dtype`` is enforced in SQL too, because SQLite
    stores every integer as signed 64-bit.
    """
    info = np.iinfo(pd.api.types.pandas_dtype(dtype).numpy_dtype)
    lo, hi = int(info.min), int(info.max)
    if hi > _SQLITE_INT_MAX:
        raise ValueError(
            f"{name}: {dtype} does not fit SQLite's signed 64-bit INTEGER; "
            "use 'Int64' with min_value=0"
        )
    if min_value is not None:
        lo = max(lo, min_value)

    sql_checks: list[str] = []
    checks: list[pa.Check] = []
    if (lo, hi) != (-_SQLITE_INT_MAX - 1, _SQLITE_INT_MAX):
        sql_checks.append(f"{name} BETWEEN {lo} AND {hi}")
    if min_value is not None:
        checks.append(pa.Check.ge(min_value))
    if allowed is not None:
        values = sorted(allowed)
        sql_checks.append(f"{name} IN ({', '.join(map(str, values))})")
        checks.append(pa.Check.isin(values))
    return Col(name, sql_type, dtype, nullable, tuple(sql_checks), tuple(checks))


def real_col(
    name: str,
    *,
    nullable: bool = False,
    value_range: tuple[float, float] | None = None,
) -> Col:
    # NaN and NULL are the same thing once loaded into pandas
    # (sqlite3 also stores NaN as NULL).
    sql_checks: tuple[str, ...] = ()
    checks: tuple[pa.Check, ...] = ()
    if value_range is not None:
        lo, hi = value_range
        sql_checks = (f"{name} BETWEEN {lo!r} AND {hi!r}",)
        checks = (pa.Check.in_range(lo, hi),)
    return Col(name, SqlType.REAL, np.float64, nullable, sql_checks, checks)


def text_col(
    name: str, *, nullable: bool = False, unique: bool = False, non_empty: bool = False
) -> Col:
    sql_checks: tuple[str, ...] = ()
    checks: tuple[pa.Check, ...] = ()
    if non_empty:
        sql_checks = (f"length({name}) > 0",)
        checks = (pa.Check.str_length(min_value=1),)
    return Col(name, SqlType.TEXT, str, nullable, sql_checks, checks, unique=unique)


def _to_gpkg_datetime(values: pd.Series) -> pd.Series:
    """GeoPackage DATETIME text, e.g. ``2024-05-01T10:00:00.250Z``."""
    text = values.dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str.slice(0, 23) + "Z"
    return text.astype(object).where(values.notna(), None)


def _parse_timestamps(values: pd.Series) -> pd.Series:
    """Naive timestamps from ISO 8601 text; text without a zone is taken as UTC."""
    parsed = pd.to_datetime(values, utc=True, format="ISO8601")
    return parsed.dt.tz_localize(None).astype("datetime64[us]")


def timestamp_col(name: str, *, nullable: bool = False, gpkg: bool = False) -> Col:
    """Timestamp stored as ISO-8601 text (SQLite has no datetime type).

    In pandas it is a naive ``datetime64`` holding UTC. ``gpkg=True`` stores it
    as a GeoPackage ``DATETIME`` (millisecond precision, ``Z`` suffix);
    otherwise as ``TEXT`` with full microsecond precision.
    """
    valid = f"julianday({name}) IS NOT NULL"
    if nullable:
        valid = f"{name} IS NULL OR {valid}"
    return Col(
        name,
        SqlType.DATETIME if gpkg else SqlType.TEXT,
        pa.Timestamp,
        nullable,
        sql_checks=(valid,),
        parse_dates=True,
        encode=_to_gpkg_datetime if gpkg else None,
    )


def point_z_col(name: str) -> Col:
    """GeoPackage Point Z geometry (GeoPackage binary bytes in pandas)."""
    return Col(
        name,
        SqlType.POINT,
        object,
        checks=(
            pa.Check(
                is_point_z_geometry,
                element_wise=True,
                error="Value must be a GeoPackage binary Point Z",
            ),
        ),
    )


def float64_blob_col(
    name: str, length: int | None, *, nullable: bool = False
) -> Col:
    """BLOB holding a little-endian float64 array; ``bytes`` in pandas.

    ``length`` is the number of float64 items; ``None`` means any count
    (the byte length only has to be divisible by 8).
    """
    itemsize = FLOAT64_BLOB_DTYPE.itemsize
    if length is None:
        sql_check = f"length({name}) % {itemsize} = 0"
        error = "Value must be bytes holding a float64 array"
    else:
        sql_check = f"length({name}) = {length * itemsize}"
        error = f"Value must be bytes holding {length} float64 values"
    return Col(
        name,
        SqlType.BLOB,
        object,
        nullable,
        sql_checks=(sql_check,),
        checks=(
            pa.Check(
                lambda value: is_float64_blob(value, length),
                element_wise=True,
                error=error,
            ),
        ),
    )


# Number of float64 items in common blobs.
VEC3 = 3  # positions, rotation vectors (axis * angle, radians), RGB colors
VEC2 = 2  # image coordinates
COV3 = 9  # 3x3 covariance matrix, row-major

# Rotation vectors are canonical: angle (vector norm) in [0, pi]. The tolerance
# absorbs rounding from conversions that land exactly on pi.
MAX_ROTATION_ANGLE = np.pi
_ROTATION_ANGLE_TOLERANCE = 1e-9


def is_rotation_vector(value: Any) -> bool:
    """Bytes holding 3 finite float64 values whose norm is at most pi."""
    if not is_float64_blob(value, VEC3):
        return False
    angle = np.linalg.norm(np.frombuffer(value, dtype=FLOAT64_BLOB_DTYPE))
    return bool(angle <= MAX_ROTATION_ANGLE + _ROTATION_ANGLE_TOLERANCE)


def rotation_vector_col(name: str, *, nullable: bool = False) -> Col:
    """Rotation vector (axis * angle, radians) rotating camera -> world.

    The angle is limited to [0, pi] so every rotation has one representation
    (apart from exactly pi). SQL can't decode floats, so only the blob length
    is enforced there; the angle is checked by pandera.
    """
    col = float64_blob_col(name, VEC3, nullable=nullable)
    angle_check = pa.Check(
        is_rotation_vector,
        element_wise=True,
        error="Rotation vector must be finite with angle (norm) <= pi",
    )
    return replace(col, checks=(*col.checks, angle_check))


# --- tables -------------------------------------------------------------------


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Col, ...]
    # Constraints spanning several columns.
    sql_checks: tuple[str, ...] = ()
    checks: tuple[pa.Check, ...] = ()
    # SQLite STRICT typing; GeoPackage feature tables can't be STRICT because
    # their geometry column has a geometry type name (e.g. POINT).
    strict: bool = True
    # Autoincrement INTEGER primary key that exists only in SQL (e.g. a
    # GeoPackage "fid"); it is not part of the DataFrame.
    primary_key: str | None = None

    def __post_init__(self) -> None:
        if self.strict:
            bad = [c.name for c in self.columns if c.sql_type not in _STRICT_TYPES]
            if bad:
                raise ValueError(f"{self.name}: STRICT tables can't hold {bad}")

    @cached_property
    def schema(self) -> pa.DataFrameSchema:
        return pa.DataFrameSchema(
            {col.name: col.pandera_column() for col in self.columns},
            checks=list(self.checks),
            name=self.name,
            strict=True,
        )

    @cached_property
    def read_dtypes(self) -> dict[str, Any]:
        return {col.name: col.dtype for col in self.columns if not col.parse_dates}

    @cached_property
    def parse_dates(self) -> list[str]:
        return [col.name for col in self.columns if col.parse_dates]

    def empty(self) -> pd.DataFrame:
        """Zero-row DataFrame with the same columns and dtypes as ``read``."""
        return pd.DataFrame({
            col.name: pd.Series(
                dtype="datetime64[us]" if col.parse_dates else col.dtype
            )
            for col in self.columns
        })

    def create_sql(self) -> str:
        parts = []
        if self.primary_key is not None:
            parts.append(
                f"{self.primary_key} INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL"
            )
        parts += [col.ddl() for col in self.columns]
        parts += [f"CHECK ({check})" for check in self.sql_checks]
        body = ",\n    ".join(parts)
        strict = " STRICT" if self.strict else ""
        return f"CREATE TABLE {self.name} (\n    {body}\n){strict}"

    def read(self, con: sqlite3.Connection) -> pd.DataFrame:
        names = ", ".join(col.name for col in self.columns)
        # Insertion order: the *_index columns refer to row positions.
        order = self.primary_key or "rowid"
        df = pd.read_sql(
            f"SELECT {names} FROM {self.name} ORDER BY {order}",
            con,
            dtype=self.read_dtypes,
        )
        for name in self.parse_dates:
            df[name] = _parse_timestamps(df[name])
        return self.schema.validate(df)

    def write(self, con: sqlite3.Connection, df: pd.DataFrame) -> None:
        """Validate ``df`` and append it to the table."""
        df = self.schema.validate(df)
        encoded = {c.name: c.encode(df[c.name]) for c in self.columns if c.encode}
        df.assign(**encoded).to_sql(self.name, con, if_exists="append", index=False)


# Parameter count of every camera model that defines one (CUSTOM/INVALID don't).
_CAMERA_PARAMETER_COUNTS = {
    m.value.key: m.value.number_of_parameters
    for m in CameraDistortionModelType
    if m.value.number_of_parameters is not None
}


def _camera_parameters_sql_check() -> str:
    itemsize = FLOAT64_BLOB_DTYPE.itemsize
    cases = " ".join(
        f"WHEN {key} THEN {count * itemsize}"
        for key, count in sorted(_CAMERA_PARAMETER_COUNTS.items())
    )
    return f"length(parameters) = CASE model {cases} ELSE length(parameters) END"


def _camera_parameters_match_model(df: pd.DataFrame) -> pd.Series:
    itemsize = FLOAT64_BLOB_DTYPE.itemsize
    ok = [
        model not in _CAMERA_PARAMETER_COUNTS
        or (
            isinstance(params, (bytes, bytearray))
            and len(params) == _CAMERA_PARAMETER_COUNTS[model] * itemsize
        )
        for model, params in zip(df["model"], df["parameters"], strict=True)
    ]
    return pd.Series(ok, index=df.index, dtype=bool)


CAMERAS = Table(
    "cameras",
    (
        text_col("name"),
        int_col("model", "Int8", allowed=CameraDistortionModelTypeKeys),
        int_col("width", "UInt32"),
        int_col("height", "UInt32"),
        float64_blob_col("parameters", None),  # length depends on model
        real_col("sensor_pixel_size", nullable=True),
    ),
    sql_checks=(_camera_parameters_sql_check(),),
    checks=(
        pa.Check(
            _camera_parameters_match_model,
            error="parameters length must match the camera model's parameter count",
        ),
    ),
)

IMAGES = Table("images", (
    text_col("name"),
    text_col("path"),
    int_col("camera_index", "UInt8"),
    float64_blob_col("position", VEC3),
    float64_blob_col("position_prior", VEC3, nullable=True),
    float64_blob_col("position_covariance", COV3, nullable=True),
    rotation_vector_col("rotation"),
    rotation_vector_col("rotation_prior", nullable=True),
    float64_blob_col("rotation_covariance", COV3, nullable=True),
    timestamp_col("timestamp", nullable=True),
    int_col("strip_index", "UInt8", nullable=True),
))


def _tie_point_columns() -> tuple[Col, ...]:
    return (
        float64_blob_col("position", VEC3),
        float64_blob_col("color", VEC3, nullable=True),
    )


def _projection_columns() -> tuple[Col, ...]:
    return (
        int_col("image_index", "UInt32"),
        int_col("tie_point_index", "Int64", min_value=0),
        float64_blob_col("position", VEC2),
    )


AUTO_TIE_POINTS = Table("auto_tie_points", _tie_point_columns())
AUTO_TIE_PROJECTIONS = Table("auto_tie_projections", _projection_columns())
MANUAL_TIE_POINTS = Table("manual_tie_points", _tie_point_columns())
MANUAL_TIE_PROJECTIONS = Table("manual_tie_projections", _projection_columns())

GCPS = Table("gcps", (
    text_col("name"),
    int_col("type", "UInt8", allowed=GcpTypeValues),
    float64_blob_col("position", VEC3),
    float64_blob_col("position_prior", VEC3),
    float64_blob_col("position_covariance", COV3),
))

GCP_PROJECTIONS = Table("gcp_projections", _projection_columns())

# Key-value block metadata. Values are text; encode structured values (e.g. as
# JSON) yourself. Keys starting with "atpackage." are reserved for the format.
METADATA = Table("metadata", (
    text_col("key", unique=True, non_empty=True),
    text_col("value"),
))

# Version of the file format (tables, conventions); bump on incompatible change.
FORMAT_VERSION = "0.2"
RESERVED_METADATA_PREFIX = "atpackage."


class MetadataKey(StrEnum):
    """Reserved metadata keys, written and read by the package itself."""

    FORMAT_VERSION = "atpackage.format_version"
    # srs_id in gpkg_spatial_ref_sys of the block CRS (world X, Y, Z), e.g. a
    # compound horizontal + vertical CRS or a geocentric one.
    SRS_ID = "atpackage.srs_id"


def validate_user_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Validate metadata supplied by the user, who can't set reserved keys."""
    df = METADATA.schema.validate(df)
    reserved = df.loc[df["key"].str.startswith(RESERVED_METADATA_PREFIX), "key"]
    if not reserved.empty:
        raise ValueError(
            f"Metadata keys starting with {RESERVED_METADATA_PREFIX!r} are "
            f"reserved: {reserved.tolist()}"
        )
    return df


# GeoPackage point layer of the image positions, derived from IMAGES for GIS
# software (which can't decode the float64 blobs). Regenerated on write (see
# atpackage.io.atpackage); never edited directly.
GIS_IMAGES = Table(
    "gis_images",
    (
        point_z_col("geom"),
        text_col("name"),
        text_col("path"),
        int_col("camera_index", "UInt8", sql_type=SqlType.SMALLINT),
        timestamp_col("timestamp", nullable=True, gpkg=True),
        int_col("strip_index", "UInt8", nullable=True, sql_type=SqlType.SMALLINT),
        real_col("position_x"),
        real_col("position_y"),
        real_col("position_z"),
        # Degrees, photogrammetric omega-phi-kappa (see atpackage.rotation).
        real_col("rotation_omega", value_range=(-180.0, 180.0)),
        real_col("rotation_phi", value_range=(-90.0, 90.0)),
        real_col("rotation_kappa", value_range=(-180.0, 180.0)),
    ),
    strict=False,
    primary_key="fid",
)

# GeoPackage point layer of the GCPs (geometry at the adjusted position),
# derived from GCPS. Regenerated on write; never edited directly.
GIS_GCPS = Table(
    "gis_gcps",
    (
        point_z_col("geom"),
        text_col("name"),
        int_col("type", "UInt8", allowed=GcpTypeValues, sql_type=SqlType.SMALLINT),
        real_col("position_x"),
        real_col("position_y"),
        real_col("position_z"),
        real_col("position_prior_x"),
        real_col("position_prior_y"),
        real_col("position_prior_z"),
    ),
    strict=False,
    primary_key="fid",
)


@dataclass(frozen=True)
class FeatureLayer:
    """A table registered as a GeoPackage feature layer."""

    table: Table
    geometry_column: str
    geometry_type: str
    identifier: str
    description: str


GIS_LAYERS: tuple[FeatureLayer, ...] = (
    FeatureLayer(
        GIS_IMAGES,
        "geom",
        "POINT",
        "Images",
        "Image projection centres with omega-phi-kappa in degrees",
    ),
    FeatureLayer(
        GIS_GCPS,
        "geom",
        "POINT",
        "GCPs",
        "Ground control and check points at their adjusted positions",
    ),
)

# Tables holding the package's own data.
TABLES: tuple[Table, ...] = (
    CAMERAS,
    IMAGES,
    AUTO_TIE_POINTS,
    AUTO_TIE_PROJECTIONS,
    MANUAL_TIE_POINTS,
    MANUAL_TIE_PROJECTIONS,
    GCPS,
    GCP_PROJECTIONS,
    METADATA,
)

# Tables computed from TABLES for display in GIS software.
GIS_TABLES: tuple[Table, ...] = tuple(layer.table for layer in GIS_LAYERS)


def create_all(
    con: sqlite3.Connection, crs: SpatialRefSys = UNDEFINED_CARTESIAN
) -> None:
    """Initialise an empty file as an atpackage GeoPackage.

    ``crs`` is the block CRS of world X, Y, Z (image and GCP positions); see
    :mod:`atpackage.crs`. It is added to ``gpkg_spatial_ref_sys`` and
    referenced from the reserved metadata key ``atpackage.srs_id``. The GIS
    layers use :func:`atpackage.crs.gis_crs`.
    """
    check_block_crs(crs)
    reserved = {
        MetadataKey.FORMAT_VERSION: FORMAT_VERSION,
        MetadataKey.SRS_ID: str(crs.srs_id),
    }

    with con:
        init_geopackage(con)
        add_spatial_ref_sys(con, crs)
        layer_crs = gis_crs(crs)
        add_spatial_ref_sys(con, layer_crs)
        for table in (*TABLES, *GIS_TABLES):
            con.execute(table.create_sql())
        for layer in GIS_LAYERS:
            register_features(
                con,
                layer.table.name,
                layer.geometry_column,
                layer.geometry_type,
                layer_crs.srs_id,
                z=1,
                identifier=layer.identifier,
                description=layer.description,
            )
        con.executemany(
            f"INSERT INTO {METADATA.name} (key, value) VALUES (?, ?)",
            [(str(key), value) for key, value in reserved.items()],
        )
