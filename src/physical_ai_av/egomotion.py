# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
import dataclasses
from typing import Self

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.spatial.transform as spt

from physical_ai_av.utils import interpolation, tf


@dataclasses.dataclass
class EgomotionState(interpolation.Interpolatable, tf.Transformable):
    pose: spt.RigidTransform = dataclasses.field(
        metadata=interpolation.Interpolatable.DEFAULT_RIGID_TRANSFORM_INTERPOLATION
        | tf.Transformable.POSE
    )
    velocity: npt.NDArray[np.float64] = dataclasses.field(
        metadata=interpolation.Interpolatable.LINEAR | tf.Transformable.VECTOR
    )
    acceleration: npt.NDArray[np.float64] = dataclasses.field(
        metadata=interpolation.Interpolatable.LINEAR | tf.Transformable.VECTOR
    )
    curvature: npt.NDArray[np.float64] = dataclasses.field(
        metadata=interpolation.Interpolatable.LINEAR
    )

    @classmethod
    def from_egomotion_df(cls, egomotion_df: pd.DataFrame) -> Self:
        return cls(
            pose=spt.RigidTransform.from_components(
                rotation=spt.Rotation.from_quat(egomotion_df[["qx", "qy", "qz", "qw"]].to_numpy()),
                translation=egomotion_df[["x", "y", "z"]].to_numpy(),
            ),
            velocity=egomotion_df[["vx", "vy", "vz"]].to_numpy(),
            acceleration=egomotion_df[["ax", "ay", "az"]].to_numpy(),
            curvature=egomotion_df[["curvature"]].to_numpy(),
        )
