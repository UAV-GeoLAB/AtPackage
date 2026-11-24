"""Containers for the tables of a block.

:class:`RawTables` holds the validated DataFrames of an :class:`AtPackage`;
:class:`Tables` is its public, read-only view. Field and property names are
the table names of :mod:`atpackage.schema.tables`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import pandas as pd


@dataclass(frozen=True, eq=False)
class RawTables:
    """The validated DataFrames of a block, one per package table.

    Owned by the block and never handed out directly: :class:`Tables` returns
    copies. ``eq=False`` because DataFrames have no single truth value.
    """

    cameras: pd.DataFrame
    images: pd.DataFrame
    auto_tie_points: pd.DataFrame
    auto_tie_projections: pd.DataFrame
    manual_tie_points: pd.DataFrame
    manual_tie_projections: pd.DataFrame
    gcps: pd.DataFrame
    gcp_projections: pd.DataFrame
    metadata: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        """Table name -> DataFrame, as the readers and writers use them."""
        return {field.name: getattr(self, field.name) for field in fields(self)}


class Tables:
    """Read-only view of a block's tables.

    Each property returns a shallow copy: with pandas copy-on-write, changing
    the returned DataFrame never changes the validated table of the block.
    Build a new :class:`AtPackage` from the changed DataFrames instead.
    """

    __slots__ = ("_raw",)

    def __init__(self, raw: RawTables) -> None:
        self._raw = raw

    @property
    def cameras(self) -> pd.DataFrame:
        return self._raw.cameras.copy(deep=False)

    @property
    def images(self) -> pd.DataFrame:
        return self._raw.images.copy(deep=False)

    @property
    def auto_tie_points(self) -> pd.DataFrame:
        return self._raw.auto_tie_points.copy(deep=False)

    @property
    def auto_tie_projections(self) -> pd.DataFrame:
        return self._raw.auto_tie_projections.copy(deep=False)

    @property
    def manual_tie_points(self) -> pd.DataFrame:
        return self._raw.manual_tie_points.copy(deep=False)

    @property
    def manual_tie_projections(self) -> pd.DataFrame:
        return self._raw.manual_tie_projections.copy(deep=False)

    @property
    def gcps(self) -> pd.DataFrame:
        return self._raw.gcps.copy(deep=False)

    @property
    def gcp_projections(self) -> pd.DataFrame:
        return self._raw.gcp_projections.copy(deep=False)

    @property
    def metadata(self) -> pd.DataFrame:
        """User metadata as a DataFrame with ``key`` and ``value`` columns."""
        return self._raw.metadata.copy(deep=False)
