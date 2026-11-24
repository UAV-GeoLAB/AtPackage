import sqlite3

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from atpackage.schema.tables import (
    CAMERAS,
    GIS_TABLES,
    IMAGES,
    METADATA,
    TABLES,
    create_all,
    int_col,
)


def blob(*values: float) -> bytes:
    return np.asarray(values, dtype="<f8").tobytes()


@pytest.fixture
def con():
    con = sqlite3.connect(":memory:")
    create_all(con)
    yield con
    con.close()


def images_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "name": ["a", "b"],
        "path": ["a.jpg", "b.jpg"],
        "camera_index": pd.array([0, 1], dtype="UInt8"),
        "position": [blob(1, 2, 3), blob(4, 5, 6)],
        "position_prior": [blob(1, 2, 3), None],
        "position_covariance": [None, None],
        "rotation": [blob(0, 0, 0), blob(0, 0, 1)],
        "rotation_prior": [None, None],
        "rotation_covariance": [None, None],
        "timestamp": pd.to_datetime(["2024-05-01T10:00:00.250", None]),
        "strip_index": pd.array([3, None], dtype="UInt8"),
    })


@pytest.mark.parametrize("table", (*TABLES, *GIS_TABLES), ids=lambda t: t.name)
def test_empty_table_reads_with_schema_dtypes(con, table):
    df = table.read(con)
    if table is METADATA:  # holds the reserved keys from the start
        assert df["key"].str.startswith("atpackage.").all()
    else:
        assert df.empty
    assert list(df.columns) == [c.name for c in table.columns]


def test_images_round_trip_keeps_nulls_and_dtypes(con):
    IMAGES.write(con, images_frame())
    df = IMAGES.read(con)

    assert df["strip_index"].dtype == "UInt8"
    assert df["strip_index"].tolist() == [3, pd.NA]
    assert df["position_prior"].tolist() == [blob(1, 2, 3), None]
    assert df["timestamp"].iloc[0] == pd.Timestamp("2024-05-01T10:00:00.250")
    assert pd.isna(df["timestamp"].iloc[1])
    assert con.execute(
        "SELECT typeof(strip_index) FROM images ORDER BY rowid"
    ).fetchall() == [("integer",), ("null",)]


def test_camera_model_accepts_negative_keys(con):
    CAMERAS.write(con, pd.DataFrame({
        "name": ["cam"],
        "model": pd.array([-2], dtype="Int8"),
        "width": pd.array([6000], dtype="UInt32"),
        "height": pd.array([4000], dtype="UInt32"),
        "parameters": [blob(1.0, 2.0)],
        "sensor_pixel_size": [np.nan],
    }))
    assert CAMERAS.read(con)["model"].tolist() == [-2]


def test_pandera_rejects_bad_blob():
    df = images_frame()
    df.loc[0, "position"] = b"\x00" * 7
    with pytest.raises(pandera.errors.SchemaError):
        IMAGES.schema.validate(df)


@pytest.mark.parametrize(
    "column, value",
    [("strip_index", 256), ("position", b"\x00" * 7), ("timestamp", "not a date")],
)
def test_sql_constraints_reject_bad_rows(con, column, value):
    row = {
        "name": "a", "path": "a.jpg", "camera_index": 0,
        "position": blob(1, 2, 3), "rotation": blob(0, 0, 0),
        column: value,
    }
    cols = ", ".join(row)
    marks = ", ".join("?" * len(row))
    with pytest.raises(sqlite3.IntegrityError):
        con.execute(f"INSERT INTO images ({cols}) VALUES ({marks})", list(row.values()))


@pytest.mark.parametrize(
    "column, value",
    [
        ("position", blob(1, 2)),
        ("rotation", blob(0, 0, 0, 1)),  # quaternion instead of rotation vector
        ("rotation_covariance", blob(*range(6))),
    ],
)
def test_blob_lengths_enforced_in_pandera(column, value):
    df = images_frame()
    df[column] = pd.Series([value, value], dtype=object)
    with pytest.raises(pandera.errors.SchemaError):
        IMAGES.schema.validate(df)


def rotvec(axis, angle: float) -> bytes:
    axis = np.asarray(axis, dtype=float)
    return blob(*(axis / np.linalg.norm(axis) * angle))


@pytest.mark.parametrize("column", ["rotation", "rotation_prior"])
@pytest.mark.parametrize(
    "value, valid",
    [
        (rotvec([1, 2, 3], 0.0), True),
        (rotvec([1, 2, 3], 1.0), True),
        (rotvec([1, 2, 3], np.pi), True),
        (rotvec([1, 2, 3], np.pi + 1e-12), True),  # rounding tolerance
        (rotvec([1, 2, 3], np.pi + 1e-6), False),
        (rotvec([0, 0, 1], 2 * np.pi - 0.1), False),  # same as -0.1 about z
        (blob(np.nan, 0, 0), False),
        (blob(np.inf, 0, 0), False),
    ],
)
def test_rotation_vector_angle(column, value, valid):
    df = images_frame()
    df[column] = pd.Series([value, value], dtype=object)
    if valid:
        IMAGES.schema.validate(df)
    else:
        with pytest.raises(pandera.errors.SchemaError):
            IMAGES.schema.validate(df)


def test_blob_lengths_enforced_in_sql(con):
    sql = "INSERT INTO images (name, path, camera_index, position, rotation)"
    con.execute(f"{sql} VALUES ('a', 'a.jpg', 0, ?, ?)", (blob(1, 2, 3), blob(0, 0, 0)))
    with pytest.raises(sqlite3.IntegrityError):
        con.execute(
            f"{sql} VALUES ('a', 'a.jpg', 0, ?, ?)", (blob(1, 2, 3), blob(0, 0, 0, 1))
        )


def cameras_frame(model: int, n_params: int) -> pd.DataFrame:
    return pd.DataFrame({
        "name": ["cam"],
        "model": pd.array([model], dtype="Int8"),
        "width": pd.array([6000], dtype="UInt32"),
        "height": pd.array([4000], dtype="UInt32"),
        "parameters": [blob(*range(n_params))],
        "sensor_pixel_size": [np.nan],
    })


@pytest.mark.parametrize(
    "model, n_params, valid",
    [(1, 4, True), (1, 5, False), (6, 12, True), (-2, 7, True), (-1, 0, True)],
)
def test_camera_parameters_match_model(con, model, n_params, valid):
    df = cameras_frame(model, n_params)
    row = df.astype(object).iloc[0].tolist()
    insert = "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)"
    if valid:
        CAMERAS.schema.validate(df)
        con.execute(insert, row)
    else:
        with pytest.raises(pandera.errors.SchemaError):
            CAMERAS.schema.validate(df)
        with pytest.raises(sqlite3.IntegrityError):
            con.execute(insert, row)


def test_uint64_is_refused():
    with pytest.raises(ValueError, match="Int64"):
        int_col("x", "UInt64")
