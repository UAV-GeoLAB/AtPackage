"""Readers and writers of photogrammetric project formats.

Each format lives in its own subpackage (e.g. :mod:`atpackage.io.atpackage`).
Readers return a :class:`PackageData`, writers take the package's tables, so
the format code never depends on :class:`atpackage.AtPackage`, which exposes
them as ``AtPackage.from_<format>()`` / ``AtPackage.to_<format>()``.
"""

from ._package_data import PackageData

__all__ = ["PackageData"]
