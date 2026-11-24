from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..schema.geopackage import UNDEFINED_CARTESIAN, SpatialRefSys


@dataclass(frozen=True)
class PackageData:
    """Format-independent contents of a package, as produced by readers.

    ``tables`` maps table names (``"cameras"``, ``"images"``, ...,
    ``"metadata"``) to DataFrames; ``metadata`` holds only user keys.
    ``crs`` is the block CRS of world X, Y, Z (see :mod:`atpackage.crs`).
    """

    tables: dict[str, pd.DataFrame]
    crs: SpatialRefSys = UNDEFINED_CARTESIAN
