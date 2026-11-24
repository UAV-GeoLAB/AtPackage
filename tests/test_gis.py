import sqlite3
import struct

import numpy as np
import pandas as pd
import pytest

from atpackage.io.atpackage._gis import (
    gis_gcps_frame,
    gis_images_frame,
    write_gis_gcps,
    write_gis_images,
)
from atpackage.schema.geopackage import (
    APPLICATION_ID,
    UNDEFINED_CARTESIAN,
    USER_VERSION,
    SpatialRefSys,
)
from atpackage.schema.tables import (
    GCPS,
    GIS_GCPS,
    GIS_IMAGES,
    GIS_TABLES,
    IMAGES,
    create_all,
)

PUWG_1992 = SpatialRefSys("ETRF2000-PL / CS92", 2180, "EPSG", 2180, "PROJCS[...]")


def blob(*values: float) -> bytes:
    return np.asarray(values, dtype="<f8").tobytes()


def images_frame() -> pd.DataFrame:
    return IMAGES.schema.validate(pd.DataFrame({
        "name": ["nadir", "south_up"],
        "path": ["a.jpg", "b.jpg"],
        "camera_index": pd.array([0, 0], dtype="UInt8"),
        "position": [blob(500000, 5800000, 300), blob(500100, 5800050, 301.5)],
        "position_prior": [None, None],
        "position_covariance": [None, None],
        # Both nadir. pi about X: image top points north (+Y) -> kappa 0.
        # pi about Y: camera y (down) -> +Y, so image top points south -> kappa 180.
        "rotation": [blob(np.pi, 0, 0), blob(0, np.pi, 0)],
        "rotation_prior": [None, None],
        "rotation_covariance": [None, None],
        "timestamp": pd.to_datetime(["2024-05-01T10:00:00.250", None]),
        "strip_index": pd.array([1, None], dtype="UInt8"),
    }))


def gcps_frame() -> pd.DataFrame:
    return GCPS.schema.validate(pd.DataFrame({
        "name": ["GCP1", "CHK1"],
        "type": pd.array([0, 3], dtype="UInt8"),
        "position": [blob(500010.0, 5800020.0, 101.0), blob(500090.0, 5800080.0, 99.0)],
        "position_prior": [blob(500010.1, 5800020.1, 101.1), blob(500090, 5800080, 99)],
        "position_covariance": [blob(*np.eye(3).ravel())] * 2,
    }))


@pytest.fixture
def con():
    con = sqlite3.connect(":memory:")
    create_all(con, PUWG_1992)
    yield con
    con.close()


def decode_point_z(geom: bytes) -> tuple[int, tuple[float, float, float]]:
    magic, version, flags, srs_id = struct.unpack_from("<2sBBi", geom)
    assert (magic, version, flags) == (b"GP", 0, 1)
    byte_order, wkb_type = struct.unpack_from("<BI", geom, 8)
    assert (byte_order, wkb_type) == (1, 1001)
    return srs_id, struct.unpack_from("<3d", geom, 13)


def test_gis_images_values():
    gis = gis_images_frame(images_frame(), PUWG_1992)
    assert list(gis.columns) == [c.name for c in GIS_IMAGES.columns]
    assert gis["position_x"].tolist() == [500000.0, 500100.0]
    assert gis["position_z"].tolist() == [300.0, 301.5]
    np.testing.assert_allclose(
        gis[["rotation_omega", "rotation_phi"]].to_numpy(), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(gis["rotation_kappa"].iloc[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(abs(gis["rotation_kappa"].iloc[1]), 180.0, atol=1e-12)
    assert gis["strip_index"].dtype == "UInt8"
    assert pd.isna(gis["strip_index"].iloc[1])
    assert decode_point_z(gis["geom"].iloc[1]) == (2180, (500100.0, 5800050.0, 301.5))


def test_gis_gcps_values():
    gis = gis_gcps_frame(gcps_frame(), PUWG_1992)
    assert list(gis.columns) == [c.name for c in GIS_GCPS.columns]
    assert gis["type"].tolist() == [0, 3]
    assert gis["position_prior_x"].tolist() == [500010.1, 500090.0]
    assert decode_point_z(gis["geom"].iloc[0]) == (2180, (500010.0, 5800020.0, 101.0))


def test_gis_frames_from_empty_tables():
    assert gis_images_frame(IMAGES.empty(), UNDEFINED_CARTESIAN).empty
    assert gis_gcps_frame(GCPS.empty(), UNDEFINED_CARTESIAN).empty


def test_file_is_a_geopackage(con):
    assert con.execute("PRAGMA application_id").fetchone() == (APPLICATION_ID,)
    assert con.execute("PRAGMA user_version").fetchone() == (USER_VERSION,)
    srs_ids = {r[0] for r in con.execute("SELECT srs_id FROM gpkg_spatial_ref_sys")}
    assert srs_ids == {-1, 0, 4326, 2180}
    extensions = con.execute(
        "SELECT table_name, column_name, extension_name, scope FROM gpkg_extensions"
        " ORDER BY column_name"
    ).fetchall()
    assert extensions == [
        ("gpkg_spatial_ref_sys", "definition_12_063", "gpkg_crs_wkt_1_1", "read-write"),
        ("gpkg_spatial_ref_sys", "epoch", "gpkg_crs_wkt_1_1", "read-write"),
    ]
    contents = con.execute(
        "SELECT table_name, data_type, srs_id FROM gpkg_contents ORDER BY table_name"
    ).fetchall()
    assert contents == [
        ("gis_gcps", "features", 2180),
        ("gis_images", "features", 2180),
    ]
    columns = con.execute(
        "SELECT table_name, column_name, geometry_type_name, srs_id, z, m"
        " FROM gpkg_geometry_columns ORDER BY table_name"
    ).fetchall()
    assert columns == [
        ("gis_gcps", "geom", "POINT", 2180, 1, 0),
        ("gis_images", "geom", "POINT", 2180, 1, 0),
    ]
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []
    assert con.execute("PRAGMA integrity_check").fetchone() == ("ok",)


@pytest.mark.parametrize("table", GIS_TABLES, ids=lambda t: t.name)
def test_gis_columns_use_geopackage_types(con, table):
    gpkg_types = {
        "BOOLEAN", "TINYINT", "SMALLINT", "MEDIUMINT", "INT", "INTEGER", "FLOAT",
        "DOUBLE", "REAL", "TEXT", "BLOB", "DATE", "DATETIME", "POINT",
    }
    info = con.execute(f"PRAGMA table_info({table.name})").fetchall()
    assert {row[2] for row in info} <= gpkg_types
    fid = info[0]
    assert (fid[1], fid[2], fid[5]) == ("fid", "INTEGER", 1)  # name, type, pk


def test_write_gis_images(con):
    write_gis_images(con, images_frame(), PUWG_1992)
    write_gis_images(con, images_frame(), PUWG_1992)  # replaces, doesn't append
    stored = GIS_IMAGES.read(con)
    pd.testing.assert_frame_equal(stored, gis_images_frame(images_frame(), PUWG_1992))
    timestamps = con.execute("SELECT timestamp FROM gis_images ORDER BY fid").fetchall()
    assert timestamps == [("2024-05-01T10:00:00.250Z",), (None,)]
    extent = con.execute(
        "SELECT min_x, min_y, max_x, max_y FROM gpkg_contents"
        " WHERE table_name = 'gis_images'"
    ).fetchone()
    assert extent == (500000.0, 5800000.0, 500100.0, 5800050.0)


def test_write_gis_gcps(con):
    write_gis_gcps(con, gcps_frame(), PUWG_1992)
    stored = GIS_GCPS.read(con)
    pd.testing.assert_frame_equal(stored, gis_gcps_frame(gcps_frame(), PUWG_1992))
    extent = con.execute(
        "SELECT min_x, min_y, max_x, max_y FROM gpkg_contents"
        " WHERE table_name = 'gis_gcps'"
    ).fetchone()
    assert extent == (500010.0, 5800020.0, 500090.0, 5800080.0)


def test_write_empty_layer_clears_extent(con):
    write_gis_gcps(con, gcps_frame(), PUWG_1992)
    write_gis_gcps(con, GCPS.empty(), PUWG_1992)
    assert GIS_GCPS.read(con).empty
    extent = con.execute(
        "SELECT min_x FROM gpkg_contents WHERE table_name = 'gis_gcps'"
    ).fetchone()
    assert extent == (None,)
