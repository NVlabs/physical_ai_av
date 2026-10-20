# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Read sparse semantic occupancy from local auxiliary sidecars."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from physical_ai_av._auxiliary import _AuxiliaryReader, _frame_timestamp


@dataclass(frozen=True)
class OccupancyFrame:
    """Hold sparse voxel indices, probabilities, labels and optional velocities.

    ``voxel_coords`` has shape (N, 3), and labels and probabilities have shape (N,).
    ``grid_origin`` and ``voxel_size`` are in meters in the reader's coordinate frame.
    Optional ``velocity_vectors`` has shape (N, 3) in meters per second.
    """

    timestamp_us: int
    voxel_coords: npt.NDArray[np.int16]
    occupancy_probs: npt.NDArray[np.float32]
    semantic_labels: npt.NDArray[np.uint8]
    grid_shape: tuple[int, int, int]
    grid_origin: npt.NDArray[np.float32]
    voxel_size: float
    velocity_vectors: npt.NDArray[np.float32] | None = None
    coordinate_frame: str | None = None
    units: Literal["m"] = "m"
    velocity_units: Literal["m/s"] = "m/s"

    @property
    def voxel_centers(self) -> npt.NDArray[np.float32]:
        """Return occupied voxel centers in meters in the stored coordinate frame."""
        return self.grid_origin + (self.voxel_coords.astype(np.float32) + 0.5) * self.voxel_size


class OccupancyReader(_AuxiliaryReader):
    """Read sparse occupancy frames from a local Zarr directory or .itar file.

    ``timestamps`` contains sorted int64 microseconds. ``sensor_metadata`` preserves
    stored attributes; ``coordinate_frame`` and ``semantic_class_names`` expose the
    stored frame and class ordering, or None when absent. Close after use, or use
    the reader as a context manager.
    """

    def __init__(self, path: str | Path, sensor_name: str | None = None) -> None:
        """Open a sidecar, requiring sensor_name when multiple sensors are present."""
        super().__init__(path, "occupancy")
        try:
            sensors = sorted(self._group.group_keys())
            if sensor_name is None:
                if len(sensors) != 1:
                    raise ValueError(f"Specify sensor_name; available sensors: {sensors}")
                sensor_name = sensors[0]
            if sensor_name not in sensors:
                raise KeyError(f"Unknown sensor {sensor_name!r}; available sensors: {sensors}")
            self.sensor_name: str = sensor_name
            self._sensor_group = self._group[sensor_name]
            self.sensor_metadata: dict[str, object] = dict(self._sensor_group.attrs)
            self.coordinate_frame: str | None = self.sensor_metadata.get("coordinate_frame")
            self.semantic_class_names: list[str] | None = self._group.attrs.get(
                "semantic_class_names"
            )
            self.timestamps: npt.NDArray[np.int64] = np.asarray(
                sorted(int(key) for key in self._sensor_group.group_keys()), dtype=np.int64
            )
        except BaseException:
            self.close()
            raise

    def get_frame(
        self,
        *,
        frame_index: int | None = None,
        timestamp_us: int | None = None,
    ) -> OccupancyFrame:
        """Load one sparse frame by index or closest preceding timestamp.

        Omit both selectors for the first frame. Timestamps are in microseconds
        and must lie within the available range. No dense voxel grid is allocated.
        """
        self._ensure_open()
        timestamp = _frame_timestamp(
            self.timestamps, frame_index=frame_index, timestamp_us=timestamp_us
        )
        frame = self._sensor_group[str(timestamp)]
        grid_shape = np.asarray(frame["grid_shape"])
        return OccupancyFrame(
            timestamp_us=timestamp,
            voxel_coords=np.asarray(frame["voxel_coords"], dtype=np.int16),
            occupancy_probs=np.asarray(frame["occupancy_probs"], dtype=np.float32),
            semantic_labels=np.asarray(frame["semantic_labels"], dtype=np.uint8),
            grid_shape=(int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])),
            grid_origin=np.asarray(frame["grid_origin"], dtype=np.float32),
            voxel_size=float(np.asarray(frame["voxel_size"]).item()),
            velocity_vectors=(
                np.asarray(frame["velocity"], dtype=np.float32) if "velocity" in frame else None
            ),
            coordinate_frame=self.coordinate_frame,
        )
