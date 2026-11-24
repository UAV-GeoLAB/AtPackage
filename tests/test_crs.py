import dataclasses
import sqlite3

import numpy as np
import pandas as pd
import pytest
from pyproj import CRS, Transformer
from pyproj.crs import CompoundCRS
from scipy.spatial.transform import Rotation

from atpackage import AtPackage
from atpackage.crs import (
    CUSTOM_SRS_ID,
    as_spatial_ref_sys,
    gis_crs,
    is_geocentric_or_topocentric,
    to_gis_coordinates,
    to_pyproj,
    world_to_ecef,
)
from atpackage.io.atpackage import read_atpackage
from atpackage.io.atpackage._gis import gis_gcps_frame, gis_images_frame
from atpackage.rotation import ecef_to_enu
from atpackage.schema.geopackage import UNDEFINED, UNDEFINED_CARTESIAN, WGS84
from atpackage.schema.tables import GCPS, IMAGES

ECEF = as_spatial_ref_sys("EPSG:4978")
CS92 = as_spatial_ref_sys("EPSG:2180")
CS92_EVRF = as_spatial_ref_sys("EPSG:2180+9651")

# Camera centre: lon 21, lat 52 (Warsaw area), 300 m above the WGS 84 ellipsoid.
LON, LAT, H = 21.0, 52.0, 300.0
_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
CAMERA_ECEF = np.array(_TO_ECEF.transform(LON, LAT, H))

# Topocentric east-north-up frame with its origin about 10 km west and 5 km
# south of the camera, on the ellipsoid. WKT1 can't express it.
ORIGIN_LON, ORIGIN_LAT = 20.854, 51.955
ENU_WKT2 = (
    'PROJCRS["Site ENU",BASEGEOGCRS["WGS 84",DATUM["World Geodetic System 1984",'
    'ELLIPSOID["WGS 84",6378137,298.257223563]]],'
    'CONVERSION["Topocentric",'
    'METHOD["Geographic/topocentric conversions",ID["EPSG",9837]],'
    f'PARAMETER["Latitude of topocentric origin",{ORIGIN_LAT},'
    'ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8834]],'
    f'PARAMETER["Longitude of topocentric origin",{ORIGIN_LON},'
    'ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",8835]],'
    'PARAMETER["Ellipsoidal height of topocentric origin",0,'
    'LENGTHUNIT["metre",1],ID["EPSG",8836]]],'
    'CS[Cartesian,3],AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1]],'
    'AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1]],'
    'AXIS["(U)",up,ORDER[3],LENGTHUNIT["metre",1]]]'
)
ENU = as_spatial_ref_sys(ENU_WKT2)
_ECEF_TO_TOPO = Transformer.from_crs("EPSG:4978", CRS.from_wkt(ENU_WKT2))
CAMERA_ENU = np.array(_ECEF_TO_TOPO.transform(*CAMERA_ECEF))

# Nadir camera with image top to the north, in local ENU: camera x -> east,
# camera y (down the image) -> south, camera z (viewing) -> down.
NADIR_NORTH_UP_ENU = Rotation.from_rotvec([np.pi, 0, 0])


def blob(*values: float) -> bytes:
    return np.asarray(values, dtype="<f8").tobytes()


def camera_rotvec_ecef(camera_to_enu: Rotation) -> np.ndarray:
    enu_to_ecef = ecef_to_enu(LON, LAT, degrees=True).inv()
    return (enu_to_ecef * camera_to_enu).as_rotvec()


def ecef_images(rotvec) -> pd.DataFrame:
    return IMAGES.schema.validate(pd.DataFrame({
        "name": ["IMG_1"],
        "path": ["a.jpg"],
        "camera_index": pd.array([0], dtype="UInt8"),
        "position": [blob(*CAMERA_ECEF)],
        "position_prior": [None],
        "position_covariance": [None],
        "rotation": [blob(*rotvec)],
        "rotation_prior": [None],
        "rotation_covariance": [None],
        "timestamp": pd.Series([pd.NaT], dtype="datetime64[us]"),
        "strip_index": pd.array([None], dtype="UInt8"),
    }))


def ecef_gcps() -> pd.DataFrame:
    return GCPS.schema.validate(pd.DataFrame({
        "name": ["GCP1"],
        "type": pd.array([0], dtype="UInt8"),
        "position": [blob(*CAMERA_ECEF)],
        "position_prior": [blob(*CAMERA_ECEF)],
        "position_covariance": [blob(*np.eye(3).ravel())],
    }))


# --- CRS definitions ----------------------------------------------------------


def test_as_spatial_ref_sys_from_epsg():
    assert (CS92.srs_id, CS92.organization, CS92.organization_coordsys_id) == (
        2180,
        "EPSG",
        2180,
    )
    assert CS92.definition.startswith('PROJCS["ETRF2000-PL / CS92"')
    assert ECEF.definition.startswith("GEOCCS[")
    assert as_spatial_ref_sys(2180) == CS92


def test_definitions_are_wkt2_and_wkt1():
    assert CS92.definition_12_063.startswith('PROJCRS["ETRF2000-PL / CS92"')
    assert ECEF.definition_12_063.startswith("GEODCRS[")
    assert CS92_EVRF.definition.startswith("COMPD_CS[")
    assert CS92_EVRF.definition_12_063.startswith("COMPOUNDCRS[")
    # WKT1 can't express a topocentric CRS; WKT2 alone defines it.
    assert ENU.definition == UNDEFINED
    assert ENU.definition_12_063.startswith('PROJCRS["Site ENU"')
    assert to_pyproj(ENU) == CRS.from_wkt(ENU_WKT2)


def test_to_pyproj_prefers_wkt2_and_falls_back_to_wkt1():
    srs = as_spatial_ref_sys("+proj=tmerc +lon_0=20.123 +ellps=GRS80")
    broken_wkt1 = dataclasses.replace(srs, definition="PROJCS[broken")
    assert to_pyproj(broken_wkt1) == to_pyproj(srs)
    wkt1_only = dataclasses.replace(srs, definition_12_063=UNDEFINED)
    assert to_pyproj(wkt1_only).equals(to_pyproj(srs), ignore_axis_order=True)
    both_broken = dataclasses.replace(broken_wkt1, definition_12_063="PROJCRS[")
    with pytest.raises(ValueError, match="Can't interpret"):
        to_pyproj(both_broken)


def test_crs_without_code_gets_custom_srs_id():
    proj = "+proj=tmerc +lat_0=0 +lon_0=20.123 +k=1 +x_0=0 +y_0=0 +ellps=GRS80"
    for srs in (as_spatial_ref_sys(proj), CS92_EVRF, ENU):
        assert (srs.srs_id, srs.organization, srs.organization_coordsys_id) == (
            CUSTOM_SRS_ID,
            "NONE",
            CUSTOM_SRS_ID,
        )
    srs = as_spatial_ref_sys(proj, srs_id=100005)
    assert (srs.srs_id, srs.organization_coordsys_id) == (100005, 100005)


def test_detection_and_gis_crs():
    assert is_geocentric_or_topocentric(ECEF)
    assert is_geocentric_or_topocentric(as_spatial_ref_sys("EPSG:7930"))  # ETRF2000
    assert is_geocentric_or_topocentric(ENU)
    assert not is_geocentric_or_topocentric(CS92)
    assert not is_geocentric_or_topocentric(CS92_EVRF)
    assert not is_geocentric_or_topocentric(UNDEFINED_CARTESIAN)
    assert gis_crs(ECEF) == WGS84
    assert gis_crs(ENU) == WGS84
    assert gis_crs(CS92) == CS92
    assert gis_crs(CS92_EVRF) == CS92  # horizontal part, with its EPSG code


def test_gis_crs_of_compound_with_custom_horizontal_part():
    horizontal = CRS("+proj=tmerc +lon_0=20.123 +ellps=GRS80")
    compound = as_spatial_ref_sys(
        CompoundCRS("Local TM + EVRF2007-PL height", [horizontal, CRS("EPSG:9651")])
    )
    layer = gis_crs(compound)
    assert compound.srs_id == CUSTOM_SRS_ID
    assert (layer.srs_id, layer.organization) == (CUSTOM_SRS_ID + 1, "NONE")
    assert to_pyproj(layer).equals(horizontal, ignore_axis_order=True)


@pytest.mark.parametrize("crs", ["EPSG:9651", "EPSG:5773"])
def test_vertical_crs_alone_is_refused(crs):
    with pytest.raises(ValueError, match="horizontal part"):
        AtPackage(crs=crs)


# --- coordinates and rotations ------------------------------------------------


def test_ecef_to_enu_axes_at_lon0_lat0():
    r = ecef_to_enu(0.0, 0.0, degrees=True)
    np.testing.assert_allclose(r.apply([1, 0, 0]), [0, 0, 1], atol=1e-15)  # up
    np.testing.assert_allclose(r.apply([0, 1, 0]), [1, 0, 0], atol=1e-15)  # east
    np.testing.assert_allclose(r.apply([0, 0, 1]), [0, 1, 0], atol=1e-15)  # north


def test_geocentric_positions_become_lon_lat_h():
    np.testing.assert_allclose(
        to_gis_coordinates(CAMERA_ECEF, ECEF), [[LON, LAT, H]], atol=1e-6
    )
    # Non-geocentric coordinates pass through unchanged.
    xyz = np.array([[637000.0, 486000.0, 300.0]])
    np.testing.assert_array_equal(to_gis_coordinates(xyz, CS92), xyz)


def test_geocentric_image_opk_is_in_local_enu():
    gis = gis_images_frame(ecef_images(camera_rotvec_ecef(NADIR_NORTH_UP_ENU)), ECEF)
    row = gis.iloc[0]
    np.testing.assert_allclose(
        [row.position_x, row.position_y, row.position_z], [LON, LAT, H], atol=1e-6
    )
    np.testing.assert_allclose(
        [row.rotation_omega, row.rotation_phi, row.rotation_kappa], 0, atol=1e-9
    )


def test_geocentric_image_kappa():
    # Turn the nadir camera 30 deg about the local vertical.
    turned = Rotation.from_euler("z", 30, degrees=True) * NADIR_NORTH_UP_ENU
    gis = gis_images_frame(ecef_images(camera_rotvec_ecef(turned)), ECEF)
    np.testing.assert_allclose(gis["rotation_kappa"].iloc[0], 30, atol=1e-9)


def test_geocentric_gcps_become_lon_lat_h():
    gis = gis_gcps_frame(ecef_gcps(), ECEF)
    np.testing.assert_allclose(
        gis[["position_prior_x", "position_prior_y", "position_prior_z"]].iloc[0],
        [LON, LAT, H],
        atol=1e-6,
    )


# --- files ------------------------------------------------------------------


def test_geocentric_package_file(tmp_path):
    path = tmp_path / "ecef.at.gpkg"
    rotvec = camera_rotvec_ecef(NADIR_NORTH_UP_ENU)
    AtPackage(
        images=ecef_images(rotvec), gcps=ecef_gcps(), crs="EPSG:4978"
    ).to_atpackage(path)

    with sqlite3.connect(path) as con:
        layer_srs = dict(
            con.execute("SELECT table_name, srs_id FROM gpkg_geometry_columns")
        )
        metadata = dict(con.execute("SELECT key, value FROM metadata"))
        stored_lon = con.execute("SELECT position_x FROM gis_images").fetchone()[0]
    con.close()
    assert layer_srs == {"gis_images": 4326, "gis_gcps": 4326}
    assert metadata["atpackage.srs_id"] == "4978"
    assert stored_lon == pytest.approx(LON, abs=1e-9)

    # The data tables keep the ECEF coordinates.
    data = read_atpackage(path)
    assert data.crs.srs_id == 4978
    position = np.frombuffer(data.tables["images"]["position"].iloc[0])
    np.testing.assert_array_equal(position, CAMERA_ECEF)


def test_compound_package_uses_horizontal_crs_in_gis(tmp_path):
    path = tmp_path / "cs92.at.gpkg"
    AtPackage(crs="EPSG:2180+9651").to_atpackage(path)
    with sqlite3.connect(path) as con:
        rows = con.execute("SELECT srs_id FROM gpkg_geometry_columns")
        layer_srs = {row[0] for row in rows}
        organization, wkt1, wkt2 = con.execute(
            "SELECT organization, definition, definition_12_063"
            " FROM gpkg_spatial_ref_sys WHERE srs_id = ?",
            (CUSTOM_SRS_ID,),
        ).fetchone()
    con.close()
    assert layer_srs == {2180}
    assert organization == "NONE"
    assert wkt1.startswith("COMPD_CS[")
    assert wkt2.startswith("COMPOUNDCRS[")
    data = read_atpackage(path)
    assert data.crs == CS92_EVRF
    assert to_pyproj(data.crs) == CRS("EPSG:2180+9651")


# --- topocentric ------------------------------------------------------------


def test_topocentric_axes_in_ecef():
    # The topocentric axes are east, north, up at the origin.
    expected = ecef_to_enu(ORIGIN_LON, ORIGIN_LAT, degrees=True).inv()
    assert (world_to_ecef(ENU) * expected.inv()).magnitude() < 1e-12
    assert world_to_ecef(ECEF).magnitude() == 0
    with pytest.raises(ValueError, match="neither"):
        world_to_ecef(CS92)


def test_topocentric_positions_become_lon_lat_h():
    np.testing.assert_allclose(
        to_gis_coordinates(CAMERA_ENU, ENU), [[LON, LAT, H]], atol=1e-6
    )


def test_topocentric_and_geocentric_blocks_give_the_same_gis_layer():
    # One physical camera, nadir and north-up, turned 30 deg about the local
    # vertical, expressed in a geocentric and in a topocentric block.
    turned = Rotation.from_euler("z", 30, degrees=True) * NADIR_NORTH_UP_ENU
    ecef_to_topo = ecef_to_enu(ORIGIN_LON, ORIGIN_LAT, degrees=True)
    rotvec_ecef = camera_rotvec_ecef(turned)
    rotvec_enu = (ecef_to_topo * Rotation.from_rotvec(rotvec_ecef)).as_rotvec()
    images_enu = ecef_images(rotvec_enu)
    images_enu["position"] = [blob(*CAMERA_ENU)]

    gis_ecef = gis_images_frame(ecef_images(rotvec_ecef), ECEF)
    gis_enu = gis_images_frame(images_enu, ENU)
    columns = ["position_x", "position_y", "position_z"]
    np.testing.assert_allclose(gis_enu[columns], gis_ecef[columns], atol=1e-6)
    angles = ["rotation_omega", "rotation_phi", "rotation_kappa"]
    np.testing.assert_allclose(gis_enu[angles], gis_ecef[angles], atol=1e-7)
    np.testing.assert_allclose(gis_enu["rotation_kappa"].iloc[0], 30, atol=1e-7)


def test_topocentric_package_file(tmp_path):
    path = tmp_path / "enu.at.gpkg"
    images = ecef_images(NADIR_NORTH_UP_ENU.as_rotvec())
    images["position"] = [blob(*CAMERA_ENU)]
    AtPackage(images=images, crs=ENU_WKT2).to_atpackage(path)

    with sqlite3.connect(path) as con:
        block_row = con.execute(
            "SELECT definition, definition_12_063 FROM gpkg_spatial_ref_sys"
            " WHERE srs_id = ?",
            (CUSTOM_SRS_ID,),
        ).fetchone()
        rows = con.execute("SELECT srs_id FROM gpkg_geometry_columns")
        layer_srs = {row[0] for row in rows}
    con.close()
    assert block_row == (UNDEFINED, ENU.definition_12_063)
    assert layer_srs == {4326}

    loaded = AtPackage.from_atpackage(path)
    assert loaded._crs == ENU
    assert to_pyproj(loaded._crs) == CRS.from_wkt(ENU_WKT2)
