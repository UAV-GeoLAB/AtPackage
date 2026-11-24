import numpy as np
import pandas as pd
import pandera.errors
import pytest
from pyproj import CRS

from atpackage.atpackage import AtPackage
from atpackage.schema.tables import CAMERAS, TABLES
from atpackage.tables import RawTables, Tables


def test_containers_cover_every_table():
    names = [table.name for table in TABLES]
    assert list(RawTables.__dataclass_fields__) == names
    assert all(isinstance(getattr(Tables, name), property) for name in names)


@pytest.mark.parametrize("table", TABLES, ids=lambda t: t.name)
def test_default_tables_are_empty_and_valid(table):
    df = getattr(AtPackage()._raw_tables, table.name)
    assert df.empty
    assert list(df.columns) == [c.name for c in table.columns]
    table.schema.validate(df)


def cameras_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "name": ["cam"],
        "model": pd.array([1], dtype="Int8"),
        "width": pd.array([6000], dtype="UInt32"),
        "height": pd.array([4000], dtype="UInt32"),
        "parameters": [np.zeros(4).tobytes()],
        "sensor_pixel_size": [np.nan],
    })


def test_supplied_table_is_copied():
    source = cameras_frame()
    pkg = AtPackage(cameras=source)
    source.loc[0, "name"] = "changed"
    assert pkg._raw_tables.cameras["name"].tolist() == ["cam"]
    CAMERAS.schema.validate(pkg._raw_tables.cameras)


def test_invalid_table_is_rejected():
    df = cameras_frame()
    df["width"] = df["width"].astype("Int64")
    with pytest.raises(pandera.errors.SchemaError):
        AtPackage(cameras=df)


@pytest.mark.parametrize("table", TABLES, ids=lambda t: t.name)
def test_every_table_is_in_the_public_view(table):
    pkg = AtPackage()
    public = getattr(pkg.tables, table.name)
    pd.testing.assert_frame_equal(public, getattr(pkg._raw_tables, table.name))
    assert public is not getattr(pkg._raw_tables, table.name)


def test_changing_a_returned_table_leaves_the_block_unchanged():
    pkg = AtPackage(cameras=cameras_frame(), metadata={"project": "x"})
    cameras = pkg.tables.cameras
    cameras.loc[0, "name"] = "changed"
    cameras["width"] = pd.array([1], dtype="UInt32")
    cameras["extra"] = 1
    metadata = pkg.tables.metadata
    metadata.loc[0, "value"] = "changed"
    assert pkg.tables.cameras["name"].tolist() == ["cam"]
    assert pkg.tables.cameras["width"].tolist() == [6000]
    assert "extra" not in pkg.tables.cameras
    assert pkg.tables.metadata["value"].tolist() == ["x"]
    CAMERAS.schema.validate(pkg._raw_tables.cameras)


def test_tables_are_read_only():
    pkg = AtPackage()
    with pytest.raises(AttributeError):
        pkg.tables.cameras = cameras_frame()
    with pytest.raises(AttributeError):
        pkg.tables = None
    with pytest.raises(AttributeError):  # frozen dataclass
        pkg._raw_tables.cameras = cameras_frame()


def test_crs_property():
    assert AtPackage().crs is None
    pkg = AtPackage(crs="EPSG:2180+9651")
    assert pkg.crs == CRS("EPSG:2180+9651")
    # A block can be rebuilt from its own properties. (Its WKT1 may lose a few
    # AUTHORITY tags of units and datums; the WKT2 is identical.)
    rebuilt = AtPackage(crs=pkg.crs)
    assert rebuilt.crs == pkg.crs
    assert rebuilt._crs.definition_12_063 == pkg._crs.definition_12_063
    assert AtPackage(crs=AtPackage().crs).crs is None
