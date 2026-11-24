from dataclasses import dataclass
from enum import Enum, IntEnum


class GcpType(IntEnum):
    XYZ_CONTROL = 0
    XY_CONTROL = 1
    Z_CONTROL = 2
    XYZ_CHECK = 3
    XY_CHECK = 4
    Z_CHECK = 5

GcpTypeNames = {x.name for x in GcpType}
GcpTypeValues = {x.value for x in GcpType}


@dataclass(frozen=True)
class CameraDistortionModel:
    key: int
    name: str
    number_of_parameters: int | None
    parameter_description: str | None
    
# TODO: Implement check for length of blob
class CameraDistortionModelType(Enum):
    CUSTOM=CameraDistortionModel(-2, "CUSTOM", None, None)
    INVALID=CameraDistortionModel(-1, "INVALID", None, None)
    SIMPLE_PINHOLE = CameraDistortionModel(0, "SIMPLE_PINHOLE", 3, "f, cx, cy")
    PINHOLE = CameraDistortionModel(1, "PINHOLE", 4, "fx, fy, cx, cy")
    SIMPLE_RADIAL = CameraDistortionModel(2, "SIMPLE_RADIAL", 4, "f, cx, cy, k")
    RADIAL = CameraDistortionModel(3, "RADIAL", 5, "f, cx, cy, k1, k2")
    OPENCV = CameraDistortionModel(
        4, "OPENCV", 8, "fx, fy, cx, cy, k1, k2, p1, p2"
    )
    OPENCV_FISHEYE = CameraDistortionModel(
        5, "OPENCV_FISHEYE", 8, "fx, fy, cx, cy, k1, k2, k3, k4"
    )
    FULL_OPENCV = CameraDistortionModel(
        6, "FULL_OPENCV", 12, "fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6"
    )
    FOV = CameraDistortionModel(7, "FOV", 5, "fx, fy, cx, cy, omega")
    SIMPLE_RADIAL_FISHEYE = CameraDistortionModel(
        8, "SIMPLE_RADIAL_FISHEYE", 4, "f, cx, cy, k"
    )
    RADIAL_FISHEYE = CameraDistortionModel(
        9, "RADIAL_FISHEYE", 5, "f, cx, cy, k1, k2"
    )
    THIN_PRISM_FISHEYE = CameraDistortionModel(
        10,
        "THIN_PRISM_FISHEYE",
        12,
        "fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1",
    )
    RAD_TAN_THIN_PRISM_FISHEYE = CameraDistortionModel(
        11,
        "RAD_TAN_THIN_PRISM_FISHEYE",
        16,
        "fx, fy, cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3",
    )
    SIMPLE_DIVISION = CameraDistortionModel(
        12, "SIMPLE_DIVISION", 4, "f, cx, cy, k"
    )
    DIVISION = CameraDistortionModel(13, "DIVISION", 5, "fx, fy, cx, cy, k")
    SIMPLE_FISHEYE = CameraDistortionModel(14, "SIMPLE_FISHEYE", 3, "f, cx, cy")
    FISHEYE = CameraDistortionModel(15, "FISHEYE", 4, "fx, fy, cx, cy")
    EUCM = CameraDistortionModel(16, "EUCM", 6, "fx, fy, cx, cy, alpha, beta")
    EQUIRECTANGULAR = CameraDistortionModel(17, "EQUIRECTANGULAR", 2, "w,h")
    SKEWED_PINHOLE = CameraDistortionModel(
        18, "SKEWED_PINHOLE", 5, "fx, fy, cx, cy, s"
    )

CameraDistortionModelTypeKeys = {x.value.key for x in CameraDistortionModelType}
CameraDistortionModelTypeValues = {x.value for x in CameraDistortionModelType}