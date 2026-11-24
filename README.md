# AtPackage

AtPackage is a format for transferring photogrammetric blocks of images between
different software, together with a Python package that reads and writes it.

> **Status:** early development. The file format and the API may still change.

## The format

An AtPackage file uses the `.at.gpkg` extension and is a valid
[OGC GeoPackage](https://www.geopackage.org/) (SQLite), so GIS software such as
QGIS can open it directly.

It holds these tables:

| Table | Content |
|---|---|
| `cameras` | Camera models (COLMAP-style distortion models) and their parameters |
| `images` | Image paths, poses, priors and covariances, timestamps, strips |
| `auto_tie_points`, `auto_tie_projections` | Automatic tie points and their image measurements |
| `manual_tie_points`, `manual_tie_projections` | Manual tie points and their image measurements |
| `gcps`, `gcp_projections` | Ground control and check points and their image measurements |
| `metadata` | Key-value block metadata: format version, block CRS and your own keys |
| `gis_images`, `gis_gcps` | Point layers for display in GIS, derived from `images` and `gcps` |

Conventions:

- **Pose:** `position` is the projection centre in world coordinates.
  `rotation` is a rotation vector (axis × angle in radians, angle ≤ π) from
  camera to world, as in Agisoft Metashape. The camera axes are x right,
  y down and z along the viewing direction.
- **CRS:** world coordinates use a single block CRS. It can be a compound
  CRS (horizontal + vertical, e.g. `EPSG:2180+9651`), a 3D CRS (geocentric
  ECEF, geographic 3D, or topocentric east-north-up at a given origin), or a
  2D CRS when the height reference is unknown. It is stored in
  `gpkg_spatial_ref_sys` as WKT2 (GeoPackage extension `gpkg_crs_wkt_1_1`),
  plus WKT1 where WKT1 can express it. The metadata key `atpackage.srs_id`
  points to it.
- **GIS layers** use WGS 84 for geocentric and topocentric blocks, the
  horizontal part for compound blocks, and the block CRS otherwise. For
  geocentric and topocentric blocks, `position_x/y/z` are longitude, latitude
  (degrees) and ellipsoidal height, and omega-phi-kappa refer to the local
  east-north-up frame at each image. Otherwise, they are the block's own
  coordinates and axes. Omega-phi-kappa are always in degrees.
- **Vectors and matrices** are stored as little-endian float64 blobs.
  Covariances are 3×3 matrices in row-major order.

## Installation

Requires Python 3.12 or newer. The package isn't on PyPI yet, so install it
from a clone of this repository:

```bash
uv pip install .   # or: pip install .
```

## Usage

```python
from atpackage import AtPackage

block = AtPackage(
    cameras=cameras_df,
    images=images_df,
    crs="EPSG:2180+9651",  # or any pyproj CRS input, e.g. a WKT2 string
    metadata={"project": "Example block"},
)
block.to_atpackage("block.at.gpkg")

block = AtPackage.from_atpackage("block.at.gpkg")
block.tables.images  # read-only view of the tables, each a DataFrame
block.crs            # pyproj.CRS, or None for an unreferenced block
```

Every table is a pandas DataFrame and is validated against its schema
(`atpackage.schema.tables`). Any table you don't pass starts empty. The block
is immutable: changing a DataFrame returned by `block.tables` doesn't change
the block, so build a new `AtPackage` from the changed tables instead.

## Development

```bash
uv sync
uv run pytest
uv run ruff check
```
