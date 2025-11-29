# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
import dataclasses
import enum
from typing import Callable, ClassVar

import numpy as np
import numpy.typing as npt
import scipy.interpolate as spi
import scipy.spatial.transform as spt


class InterpolationMethod(enum.StrEnum):
    """Supported interpolation methods for vectors and rotations."""

    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"
    SLERP = "slerp"
    ROTATION_SPLINE = "rotation_spline"


@dataclasses.dataclass
class RigidTransformInterpolationMethod:
    """Specifies an interpolation method for `RigidTransform`s."""

    rotation: InterpolationMethod = InterpolationMethod.SLERP
    translation: InterpolationMethod = InterpolationMethod.CUBIC_SPLINE


@dataclasses.dataclass
class RigidTransformInterpolator:
    """Interpolator for `RigidTransform`s."""

    timestamps: npt.NDArray[np.int64]
    values: spt.RigidTransform
    interpolation_method: RigidTransformInterpolationMethod = dataclasses.field(
        default_factory=RigidTransformInterpolationMethod
    )

    def __post_init__(self) -> None:
        """Constructs interpolants (using relative timestamps for numerics)."""
        self.relative_timestamps = self.timestamps - self.timestamps[0]
        self.interpolants = {
            "rotation": create_interpolant(
                self.interpolation_method.rotation,
                self.relative_timestamps,
                self.values.rotation,
            ),
            "translation": create_interpolant(
                self.interpolation_method.translation,
                self.relative_timestamps,
                self.values.translation,
            ),
        }

    def __call__(self, timestamp: npt.ArrayLike) -> spt.RigidTransform:
        """Interpolates the `values` at `timestamp`."""
        return spt.RigidTransform.from_components(
            rotation=self.interpolants["rotation"](timestamp - self.timestamps[0]),
            translation=self.interpolants["translation"](timestamp - self.timestamps[0]),
        )


def create_interpolant(
    interpolation_method: InterpolationMethod | RigidTransformInterpolationMethod,
    timestamps: npt.NDArray[np.int64],
    values: npt.ArrayLike | spt.Rotation | spt.RigidTransform,
) -> Callable[[npt.ArrayLike], npt.NDArray[np.floating] | spt.Rotation | spt.RigidTransform]:
    """Creates an interpolator (notably with extrapolation disallowed)."""
    if interpolation_method == InterpolationMethod.LINEAR:
        if not isinstance(values, np.ndarray):
            raise ValueError(f"Expected `values` to be a `NDArray`, got {type(values)}.")
        linear = spi.make_interp_spline(timestamps, values, k=1)
        linear.extrapolate = False
        return linear
    elif interpolation_method == InterpolationMethod.CUBIC_SPLINE:
        if not isinstance(values, np.ndarray):
            raise ValueError(f"Expected `values` to be a `NDArray`, got {type(values)}.")
        return spi.CubicSpline(timestamps, values, extrapolate=False)
    elif interpolation_method == InterpolationMethod.SLERP:
        if not isinstance(values, spt.Rotation):
            raise ValueError(f"Expected `values` to be a `Rotation`, got {type(values)}.")
        return spt.Slerp(timestamps, values)
    elif interpolation_method == InterpolationMethod.ROTATION_SPLINE:
        if not isinstance(values, spt.Rotation):
            raise ValueError(f"Expected `values` to be a `Rotation`, got {type(values)}.")
        return spt.RotationSpline(timestamps, values)
    elif isinstance(interpolation_method, RigidTransformInterpolationMethod):
        if not isinstance(values, spt.RigidTransform):
            raise ValueError(f"Expected `values` to be a `RigidTransform`, got {type(values)}.")
        return RigidTransformInterpolator(timestamps, values, interpolation_method)
    raise ValueError(f"Unknown {interpolation_method=}.")


@dataclasses.dataclass
class Interpolator[T: Interpolatable]:
    """Interpolator for dataclasses with interpolation method specified by field metadata."""

    timestamps: npt.NDArray[np.int64]
    values: T

    def __post_init__(self) -> None:
        """Constructs interpolants (using relative timestamps for numerics)."""
        self.value_type = type(self.values)
        self.relative_timestamps = self.timestamps - self.timestamps[0]

        def _create_interpolant(field):
            field_values = getattr(self.values, field.name)
            if isinstance(field_values, Interpolatable):
                return field_values.create_interpolator(self.relative_timestamps)
            if Interpolatable.INTERPOLATION_METHOD_KEY not in field.metadata:
                raise ValueError(
                    f"Missing interpolation method for {self.value_type.__name__}.{field.name}. "
                    "If the field value type isn't already `Interpolatable`, "
                    "then the interpolation method must be specified as dataclass field metadata."
                )
            interpolation_method = field.metadata[Interpolatable.INTERPOLATION_METHOD_KEY]
            return create_interpolant(interpolation_method, self.relative_timestamps, field_values)

        self.interpolants = {
            field.name: _create_interpolant(field) for field in dataclasses.fields(self.value_type)
        }

    def __call__(self, timestamp: npt.ArrayLike) -> T:
        """Interpolates the `values` at `timestamp`."""
        return self.value_type(
            **{
                field.name: self.interpolants[field.name](timestamp - self.timestamps[0])
                for field in dataclasses.fields(self.value_type)
            }
        )

    @property
    def time_range(self) -> tuple[int, int]:
        """Returns the time range of the interpolator."""
        return self.timestamps[0], self.timestamps[-1]

    def __repr__(self) -> str:
        """Returns a concise repr of the interpolator including the value type and time range."""
        return f"Interpolator[{self.value_type.__name__}](time_range=[{self.time_range}])"


class Interpolatable:
    """Base class for enabling interpolatable dataclasses."""

    INTERPOLATION_METHOD_KEY: ClassVar[str] = "interpolation_method"
    LINEAR: ClassVar[dict[str, InterpolationMethod]] = {
        INTERPOLATION_METHOD_KEY: InterpolationMethod.LINEAR
    }
    CUBIC_SPLINE: ClassVar[dict[str, InterpolationMethod]] = {
        INTERPOLATION_METHOD_KEY: InterpolationMethod.CUBIC_SPLINE
    }
    SLERP: ClassVar[dict[str, InterpolationMethod]] = {
        INTERPOLATION_METHOD_KEY: InterpolationMethod.SLERP
    }
    ROTATION_SPLINE: ClassVar[dict[str, InterpolationMethod]] = {
        INTERPOLATION_METHOD_KEY: InterpolationMethod.ROTATION_SPLINE
    }
    DEFAULT_RIGID_TRANSFORM_INTERPOLATION: ClassVar[
        dict[str, RigidTransformInterpolationMethod]
    ] = {INTERPOLATION_METHOD_KEY: RigidTransformInterpolationMethod()}

    def create_interpolator(self, timestamps: npt.NDArray[np.int64]) -> Interpolator:
        """Creates an `Interpolator` for this `Interpolatable` instance."""
        return Interpolator(timestamps, self)
