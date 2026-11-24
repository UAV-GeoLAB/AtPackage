from __future__ import annotations

import os
from collections.abc import Mapping

import pandas as pd
from pyproj import CRS

from .crs import CrsLike, as_spatial_ref_sys, check_block_crs, to_pyproj
from .io import atpackage as atpackage_io
from .schema.geopackage import UNDEFINED_CARTESIAN
from .schema.tables import (
    AUTO_TIE_POINTS,
    AUTO_TIE_PROJECTIONS,
    CAMERAS,
    GCP_PROJECTIONS,
    GCPS,
    IMAGES,
    MANUAL_TIE_POINTS,
    MANUAL_TIE_PROJECTIONS,
    METADATA,
    Table,
    validate_user_metadata,
)
from .tables import RawTables, Tables


def _init_table(table: Table, df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return table.empty()
    return table.schema.validate(df.copy())


def _init_metadata(metadata: pd.DataFrame | Mapping[str, str] | None) -> pd.DataFrame:
    if metadata is None:
        return METADATA.empty()
    if isinstance(metadata, Mapping):
        metadata = pd.DataFrame({
            "key": pd.Series(list(metadata.keys()), dtype="str"),
            "value": pd.Series(list(metadata.values()), dtype="str"),
        })
    return validate_user_metadata(metadata.copy())


class AtPackage:
    """In-memory photogrammetric project: one DataFrame per package table."""

    def __init__(
        self,
        *,
        cameras: pd.DataFrame | None = None,
        images: pd.DataFrame | None = None,
        auto_tie_points: pd.DataFrame | None = None,
        auto_tie_projections: pd.DataFrame | None = None,
        manual_tie_points: pd.DataFrame | None = None,
        manual_tie_projections: pd.DataFrame | None = None,
        gcps: pd.DataFrame | None = None,
        gcp_projections: pd.DataFrame | None = None,
        metadata: pd.DataFrame | Mapping[str, str] | None = None,
        crs: CrsLike | None = None,
    ) -> None:
        """Every table is optional; missing tables start empty.

        Supplied DataFrames are copied and validated against their schema.
        ``metadata`` is key-value block metadata, as a ``{key: value}`` mapping
        or a DataFrame with ``key`` and ``value`` columns; keys starting with
        ``"atpackage."`` are reserved. ``crs`` is the block CRS of world X, Y,
        Z, as a :class:`SpatialRefSys` or anything pyproj accepts: a compound
        CRS such as ``"EPSG:2180+9651"`` (or a ``pyproj.crs.CompoundCRS``), a
        3D CRS such as geocentric ``"EPSG:4978"`` or a topocentric WKT2, or a
        2D CRS when the height reference is unknown. ``None`` (the default)
        leaves the block unreferenced. See :mod:`atpackage.crs`.

        The tables are read back through :attr:`tables`, the CRS through
        :attr:`crs`.
        """
        self._raw_tables = RawTables(
            cameras=_init_table(CAMERAS, cameras),
            images=_init_table(IMAGES, images),
            auto_tie_points=_init_table(AUTO_TIE_POINTS, auto_tie_points),
            auto_tie_projections=_init_table(
                AUTO_TIE_PROJECTIONS, auto_tie_projections
            ),
            manual_tie_points=_init_table(MANUAL_TIE_POINTS, manual_tie_points),
            manual_tie_projections=_init_table(
                MANUAL_TIE_PROJECTIONS, manual_tie_projections
            ),
            gcps=_init_table(GCPS, gcps),
            gcp_projections=_init_table(GCP_PROJECTIONS, gcp_projections),
            metadata=_init_metadata(metadata),
        )
        self._tables = Tables(self._raw_tables)
        self._crs = UNDEFINED_CARTESIAN if crs is None else as_spatial_ref_sys(crs)
        check_block_crs(self._crs)

    # --- read-only access -------------------------------------------------

    @property
    def tables(self) -> Tables:
        """Read-only view of the tables: ``block.tables.cameras``, ..."""
        return self._tables

    @property
    def crs(self) -> CRS | None:
        """Block CRS of world X, Y, Z; ``None`` if the block is unreferenced."""
        return to_pyproj(self._crs)

    # --- io ---------------------------------------------------------------

    @classmethod
    def from_atpackage(cls, path: str | os.PathLike[str]) -> AtPackage:
        """Load a package from an ``.at.gpkg`` file."""
        data = atpackage_io.read_atpackage(path)
        return cls(**data.tables, crs=data.crs)

    def to_atpackage(
        self, path: str | os.PathLike[str], *, overwrite: bool = False
    ) -> None:
        """Save the package to an ``.at.gpkg`` file (a GeoPackage)."""
        atpackage_io.write_atpackage(
            path,
            self._raw_tables.as_dict(),
            self._crs,
            overwrite=overwrite,
        )
