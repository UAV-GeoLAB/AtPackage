"""The native atpackage format: an OGC GeoPackage with the ``.at.gpkg`` extension.

Holds the package tables (see :mod:`atpackage.schema.tables`) plus GeoPackage
point layers of images and GCPs for display in GIS software.
"""

from .read import read_atpackage
from .write import FILE_EXTENSION, write_atpackage

__all__ = ["FILE_EXTENSION", "read_atpackage", "write_atpackage"]
