# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
import dataclasses
import enum
from typing import ClassVar, Self

import numpy as np
import scipy.spatial.transform as spt

from physical_ai_av.utils import interpolation


@dataclasses.dataclass(frozen=True)
class FrameInfo:
    """Dataclass for communicating coordinate frame and timestamp information.

    Inspired by ROS `std_msgs/Header` and the ROS message convention of `<MessageType>Stamped`.

    Attributes:
        frame_id (`str`): The ID of the frame the data is associated with.
        timestamp (`int | str | None`): Timestamp information, depending on type:

            - If `None`, implies static (i.e., non-time-varying) data.
            - If `int`, represents a timestamp (in microseconds).
            - If `str`, indicates some custom format/processing required for the data.
    """

    frame_id: str
    timestamp: int | str | None = None

    def __lt__(self, other: Self) -> bool:
        """Compares two `FrameInfo`s by `timestamp`, provided their `frame_id`s match."""
        if self.frame_id != other.frame_id:
            raise ValueError(
                f"Nonmatching {self.frame_id=} and {other.frame_id=} cannot be compared."
            )
        if not isinstance(self.timestamp, int) or not isinstance(other.timestamp, int):
            raise ValueError(
                f"{self.frame_id=} and {other.frame_id=} cannot be compared; "
                "only integer timestamps can be compared."
            )
        return self.timestamp < other.timestamp


@dataclasses.dataclass(frozen=True)
class FrameTransform:
    """Specifies a transform between two (possibly timestamped) coordinate frames.

    Attributes:
        target_frame_info (`FrameInfo`): The target frame for coordinate conversion.
        source_frame_info (`FrameInfo`): The source frame for coordinate conversion.
        tf_target_source (`spt.RigidTransform`): Transform that converts coordinates from the source
            frame to the target frame, i.e.,
            `coordinates_target = tf_target_source.apply(coordinates_source)`.
    """

    target_frame_info: FrameInfo
    source_frame_info: FrameInfo
    tf_target_source: spt.RigidTransform

    def check(self, frame_info: FrameInfo) -> None:
        """Checks that the `FrameTransform` may be applied to data with the given `frame_info`."""
        if self.source_frame_info != frame_info:
            raise ValueError(f"{self.source_frame_info=} does not match {frame_info=}.")


class TransformTree:
    """A transform tree for keeping track of multiple coordinate frames.

    Inspired by the ROS `tf`/`tf2` libraries.

    Attributes:
        parent (`dict[str, tuple[str, spt.RigidTransform | interpolation.RigidTransformInterpolator]]`):
            Maps frames added to the transform tree to their parent frame and the transform (which
            may be time-varying) that converts coordinates from the child frame to the parent frame,
            i.e., `coordinates_parent = tf_parent_child @ coordinates_child`.
        root_frame_id (`str`): The ID of the root frame of the transform tree; assumed to be static.
    """

    def __init__(self, root_frame_id: str = "anchor") -> None:
        self.parent = {}
        self.root_frame_id = root_frame_id

    def lookup_transform(
        self,
        target_frame_info: FrameInfo,
        source_frame_info: FrameInfo,
    ) -> FrameTransform:
        """Computes the transform between a source and target frame.

        Args:
            target_frame_info (`FrameInfo`): The target frame for coordinate conversion.
            source_frame_info (`FrameInfo`): The source frame for coordinate conversion.

        Returns:
            `FrameTransform`: The transform (with associated frame information) that specifies
                coordinate transformation from the source frame to the target frame.
        """
        # TODO(eschmerling): optimize case where you don't need to go back to root.
        tf_root_target = self._compute_tf_root_frame(target_frame_info)
        tf_root_source = self._compute_tf_root_frame(source_frame_info)
        return FrameTransform(
            target_frame_info,
            source_frame_info,
            tf_root_target.inv() @ tf_root_source,
        )

    def add_transform(
        self,
        parent_frame_id: str,
        child_frame_id: str,
        tf_parent_child: spt.RigidTransform | interpolation.RigidTransformInterpolator,
    ) -> None:
        """Adds a transform between a frame and a child frame to the transform tree.

        Args:
            parent_frame_id (`str`): The ID of the parent frame, which may already be in the
                transform tree.
            child_frame_id (`str`): The ID of the child frame, which must not already be in the
                transform tree.
            tf_parent_child (`spt.RigidTransform | interpolation.RigidTransformInterpolator`): The
                transform (which may be time-varying) that converts coordinates from the child frame
                to the parent frame, i.e.,
                `coordinates_parent = tf_parent_child @ coordinates_child`.
        """
        if child_frame_id in self.parent:
            raise ValueError(f"{child_frame_id=} is already in the transform tree.")
        self.parent[child_frame_id] = (parent_frame_id, tf_parent_child)

    def _compute_tf_root_frame(self, frame_info: FrameInfo):
        """Computes the transform to the root frame from the given frame."""
        frame_id = frame_info.frame_id
        tf_root_frame = spt.RigidTransform.identity()
        while frame_id != self.root_frame_id:
            frame_id, tf_parent_frame = self.parent[frame_id]
            if not isinstance(tf_parent_frame, spt.RigidTransform):
                if isinstance(tf_parent_frame, interpolation.RigidTransformInterpolator):
                    tf_parent_frame = tf_parent_frame(frame_info.timestamp)
                else:
                    raise ValueError(f"Unknown transform type {type(tf_parent_frame)=}.")
            tf_root_frame = tf_parent_frame @ tf_root_frame
        return tf_root_frame


class TransformableType(enum.StrEnum):
    """Supported transformable types."""

    POINT = "point"
    POSE = "pose"
    VECTOR = "vector"


class Transformable:
    """Base class for enabling transformable dataclasses."""

    TRANSFORMABLE_TYPE_KEY: ClassVar[str] = "transformable_type"
    POINT: ClassVar[dict[str, TransformableType]] = {
        TRANSFORMABLE_TYPE_KEY: TransformableType.POINT
    }
    POSE: ClassVar[dict[str, TransformableType]] = {TRANSFORMABLE_TYPE_KEY: TransformableType.POSE}
    VECTOR: ClassVar[dict[str, TransformableType]] = {
        TRANSFORMABLE_TYPE_KEY: TransformableType.VECTOR
    }

    def transform(self, rigid_transform: spt.RigidTransform) -> Self:
        """Transforms this `Transformable` instance by the given `rigid_transform`."""

        def _transform_field(field):
            field_value = getattr(self, field.name)
            if isinstance(field_value, Transformable):
                return field_value.transform(rigid_transform)
            transformable_type = field.metadata[Transformable.TRANSFORMABLE_TYPE_KEY]
            if transformable_type == TransformableType.POINT:
                if not isinstance(field_value, np.ndarray):
                    raise ValueError(
                        f"Expected {field.name} to be a `NDArray`, got {type(field_value)}."
                    )
                return rigid_transform.apply(field_value)
            elif transformable_type == TransformableType.POSE:
                if not isinstance(field_value, spt.RigidTransform):
                    raise ValueError(
                        f"Expected {field.name} to be a `RigidTransform`, got {type(field_value)}."
                    )
                return rigid_transform * field_value
            elif transformable_type == TransformableType.VECTOR:
                if not isinstance(field_value, np.ndarray):
                    raise ValueError(
                        f"Expected {field.name} to be a `NDArray`, got {type(field_value)}."
                    )
                return rigid_transform.rotation.apply(field_value)
            raise ValueError(f"Unknown {transformable_type=}.")

        return dataclasses.replace(
            self,
            **{
                field.name: _transform_field(field)
                for field in dataclasses.fields(self)
                if issubclass(field.type, Transformable)
                or Transformable.TRANSFORMABLE_TYPE_KEY in field.metadata
            },
        )

    def transform_frame(self, frame_transform: FrameTransform, skip_check: bool = False) -> Self:
        """Transforms this `Transformable` instance by the given `frame_transform`."""
        if not skip_check:
            frame_transform.check(self.frame_info)
        return dataclasses.replace(
            self.transform(frame_transform.tf_target_source),
            frame_info=frame_transform.target_frame_info,
        )
