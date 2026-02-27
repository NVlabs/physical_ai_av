import dataclasses
from typing import Self

import pandas as pd
import scipy.spatial.transform as spt

from physical_ai_av.utils import camera_models as cm


@dataclasses.dataclass
class CameraIntrinsics:
    camera_models: dict[str, cm.CameraModel]

    @classmethod
    def from_intrinsics_df(
        cls,
        intrinsics_df: pd.DataFrame,
        CameraModelClass: type[cm.CameraModel] = cm.FThetaCameraModel,
    ) -> Self:
        return cls(
            {
                camera_id: CameraModelClass.from_camera_row(intrinsics_df.loc[camera_id])
                for camera_id in intrinsics_df.index
            }
        )


@dataclasses.dataclass
class SensorExtrinsics:
    sensor_poses: dict[str, spt.RigidTransform]

    @classmethod
    def from_extrinsics_df(cls, extrinsics_df: pd.DataFrame) -> Self:
        return cls(
            sensor_poses={
                sensor_id: spt.RigidTransform.from_components(
                    rotation=spt.Rotation.from_quat(
                        extrinsics_df.loc[sensor_id, ["qx", "qy", "qz", "qw"]].to_numpy(copy=True)
                    ),
                    translation=extrinsics_df.loc[sensor_id, ["x", "y", "z"]].to_numpy(copy=True),
                )
                for sensor_id in extrinsics_df.index
            },
        )


@dataclasses.dataclass
class VehicleDimensions:
    length: float
    width: float
    height: float
    rear_axle_to_bbox_center: float
    wheelbase: float
    track_width: float

    @classmethod
    def from_dimensions_df(cls, dimensions_df: pd.DataFrame) -> Self:
        return cls(
            length=dimensions_df.loc["length"].item(),
            width=dimensions_df.loc["width"].item(),
            height=dimensions_df.loc["height"].item(),
            rear_axle_to_bbox_center=dimensions_df.loc["rear_axle_to_bbox_center"].item(),
            wheelbase=dimensions_df.loc["wheelbase"].item(),
            track_width=dimensions_df.loc["track_width"].item(),
        )
