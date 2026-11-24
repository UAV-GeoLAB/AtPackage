import sqlite3

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from atpackage import AtPackage
from atpackage.crs import CUSTOM_SRS_ID, as_spatial_ref_sys
from atpackage.io.atpackage import FILE_EXTENSION, read_atpackage, write_atpackage
from atpackage.schema.geopackage import SpatialRefSys
from atpackage.schema.tables import TABLES

# ETRF2000-PL / CS92 + EVRF2007-PL height: compound, no EPSG code of its own.
CS92_EVRF = as_spatial_ref_sys("EPSG:2180+9651")


def blob(*values: float) -> bytes:
    return np.asarray(values, dtype="<f8").tobytes()


def sample_package() -> AtPackage:
    n = 50  # enough rows to catch reordering
    return AtPackage(
        cameras=pd.DataFrame({
            "name": ["cam0", "cam1"],
            "model": pd.array([1, -2], dtype="Int8"),
            "width": pd.array([6000, 4000], dtype="UInt32"),
            "height": pd.array([4000, 3000], dtype="UInt32"),
            "parameters": [blob(1, 2, 3, 4), blob(1.5)],
            "sensor_pixel_size": [3.76e-6, np.nan],
        }),
        images=pd.DataFrame({
            "name": [f"IMG_{i:03d}" for i in range(n)],
            "path": [f"img/{i}.jpg" for i in range(n)],
            "camera_index": pd.array([i % 2 for i in range(n)], dtype="UInt8"),
            "position": [blob(637000 + i, 486000, 300) for i in range(n)],
            "position_prior": [blob(637000, 486000, 300)] + [None] * (n - 1),
            "position_covariance": [None] * n,
            "rotation": [blob(np.pi, 0, 0)] * n,
            "rotation_prior": [None] * n,
            "rotation_covariance": [blob(*np.eye(3).ravel())] * n,
            "timestamp": pd.to_datetime(
                ["2024-05-01T10:00:00.123456"] * (n - 1) + [None]
            ),
            "strip_index": pd.array([i // 10 for i in range(n)], dtype="UInt8"),
        }),
        gcps=pd.DataFrame({
            "name": ["GCP1"],
            "type": pd.array([0], dtype="UInt8"),
            "position": [blob(637050, 486010, 100)],
            "position_prior": [blob(637050, 486010, 100)],
            "position_covariance": [blob(*np.eye(3).ravel())],
        }),
        gcp_projections=pd.DataFrame({
            "image_index": pd.array([3, 1], dtype="UInt32"),
            "tie_point_index": pd.array([0, 0], dtype="Int64"),
            "position": [blob(10.5, 20.5), blob(30.5, 40.5)],
        }),
        crs=CS92_EVRF,
        metadata={"project": "Warsaw 2024", "operator": "survey team"},
    )


def test_round_trip(tmp_path):
    original = sample_package()
    path = tmp_path / f"block{FILE_EXTENSION}"
    original.to_atpackage(path)
    loaded = AtPackage.from_atpackage(path)

    for name, df in original._raw_tables.as_dict().items():
        pd.testing.assert_frame_equal(loaded._raw_tables.as_dict()[name], df, obj=name)
    assert loaded._crs == CS92_EVRF


def test_written_file_has_gis_layers(tmp_path):
    path = tmp_path / "block.at.gpkg"
    sample_package().to_atpackage(path)
    with sqlite3.connect(path) as con:
        assert con.execute("SELECT count(*) FROM gis_images").fetchone() == (50,)
        assert con.execute("SELECT count(*) FROM gis_gcps").fetchone() == (1,)
    assert list(tmp_path.iterdir()) == [path]  # temporary file removed


def test_empty_package_round_trip(tmp_path):
    path = tmp_path / "empty.at.gpkg"
    AtPackage().to_atpackage(path)
    data = read_atpackage(path)
    assert set(data.tables) == {t.name for t in TABLES}
    assert all(df.empty for df in data.tables.values())
    assert data.crs.srs_id == -1


@pytest.mark.parametrize("name", ["block.gpkg", "block.at", "block.at.gpkg.bak"])
def test_wrong_extension_is_refused(tmp_path, name):
    with pytest.raises(ValueError, match=r"\.at\.gpkg"):
        AtPackage().to_atpackage(tmp_path / name)


def test_existing_file_needs_overwrite(tmp_path):
    path = tmp_path / "block.at.gpkg"
    AtPackage().to_atpackage(path)
    with pytest.raises(FileExistsError):
        sample_package().to_atpackage(path)
    sample_package().to_atpackage(path, overwrite=True)
    assert len(AtPackage.from_atpackage(path).tables.images) == 50


def test_failed_write_keeps_existing_file(tmp_path):
    path = tmp_path / "block.at.gpkg"
    sample_package().to_atpackage(path)
    bad = {"images": pd.DataFrame({"name": ["x"]})}  # fails validation
    with pytest.raises(Exception):  # noqa: B017
        write_atpackage(path, bad, overwrite=True)
    assert len(AtPackage.from_atpackage(path).tables.images) == 50
    assert list(tmp_path.iterdir()) == [path]


def test_unknown_table_is_refused(tmp_path):
    with pytest.raises(ValueError, match="Unknown tables"):
        write_atpackage(tmp_path / "x.at.gpkg", {"photos": pd.DataFrame()})


def test_reading_other_files_fails(tmp_path):
    not_sqlite = tmp_path / "a.at.gpkg"
    not_sqlite.write_text("hello")
    with pytest.raises(ValueError, match="not an atpackage"):
        read_atpackage(not_sqlite)

    plain_gpkg = tmp_path / "b.gpkg"
    with sqlite3.connect(plain_gpkg) as con:
        con.execute("PRAGMA application_id = 1196444487")
        con.execute("CREATE TABLE something (x)")
    con.close()
    with pytest.raises(ValueError, match="missing"):
        read_atpackage(plain_gpkg)

    with pytest.raises(FileNotFoundError):
        read_atpackage(tmp_path / "missing.at.gpkg")


def test_file_stores_crs_and_version_in_metadata(tmp_path):
    path = tmp_path / "block.at.gpkg"
    sample_package().to_atpackage(path)
    with sqlite3.connect(path) as con:
        rows = dict(con.execute("SELECT key, value FROM metadata").fetchall())
        srs_ids = {r[0] for r in con.execute("SELECT srs_id FROM gpkg_spatial_ref_sys")}
        layer_srs = con.execute(
            "SELECT srs_id FROM gpkg_geometry_columns WHERE table_name = 'gis_images'"
        ).fetchone()
    con.close()
    assert rows == {
        "atpackage.format_version": "0.2",
        "atpackage.srs_id": str(CUSTOM_SRS_ID),
        "project": "Warsaw 2024",
        "operator": "survey team",
    }
    assert {CUSTOM_SRS_ID, 2180} <= srs_ids
    assert layer_srs == (2180,)  # the GIS layers use the horizontal part


def test_user_metadata_excludes_reserved_keys_after_read(tmp_path):
    path = tmp_path / "block.at.gpkg"
    sample_package().to_atpackage(path)
    metadata = AtPackage.from_atpackage(path).tables.metadata
    assert dict(zip(metadata["key"], metadata["value"], strict=True)) == {
        "project": "Warsaw 2024",
        "operator": "survey team",
    }


def test_metadata_accepts_dataframe():
    df = pd.DataFrame({"key": ["a"], "value": ["1"]})
    assert AtPackage(metadata=df).tables.metadata["value"].tolist() == ["1"]


@pytest.mark.parametrize(
    "metadata",
    [
        {"atpackage.format_version": "9"},  # reserved
        {"": "empty key"},
        pd.DataFrame({"key": ["a", "a"], "value": ["1", "2"]}),  # duplicate
    ],
)
def test_invalid_metadata_is_refused(metadata):
    with pytest.raises((ValueError, pandera.errors.SchemaError)):
        AtPackage(metadata=metadata)


def test_unsupported_format_version_is_refused(tmp_path):
    path = tmp_path / "block.at.gpkg"
    AtPackage().to_atpackage(path)
    with sqlite3.connect(path) as con:
        con.execute(
            "UPDATE metadata SET value = '99.0'"
            " WHERE key = 'atpackage.format_version'"
        )
    con.close()
    with pytest.raises(ValueError, match="99.0 is not supported"):
        read_atpackage(path)


def test_conflicting_srs_id_is_refused(tmp_path):
    local_cs = (
        'LOCAL_CS["My local grid",LOCAL_DATUM["Site",0],UNIT["metre",1],'
        'AXIS["X",EAST],AXIS["Y",NORTH]]'
    )
    clash = SpatialRefSys("My local grid", 4326, "LOCAL", 1, local_cs)
    with pytest.raises(ValueError, match="srs_id 4326 is already used"):
        AtPackage(crs=clash).to_atpackage(tmp_path / "x.at.gpkg")
    assert list(tmp_path.iterdir()) == []


def test_same_authority_srs_reuses_existing_row(tmp_path):
    # WGS 84 is predefined in every GeoPackage; a re-definition is accepted.
    wgs84 = SpatialRefSys("WGS 84", 4326, "epsg", 4326, "GEOGCRS[...]")
    path = tmp_path / "x.at.gpkg"
    AtPackage(crs=wgs84).to_atpackage(path)
    assert read_atpackage(path).crs.srs_id == 4326
