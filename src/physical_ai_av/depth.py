# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Read camera ray-distance depth from local auxiliary sidecars."""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import av
import numpy as np
import numpy.typing as npt

from physical_ai_av._auxiliary import _AuxiliaryReader, _frame_timestamp


@dataclass(frozen=True)
class DepthFrame:
    """Hold one float32 depth image in meters along camera rays.

    ``valid_mask`` marks finite, strictly positive depths. Values remain unchanged
    unless the reader's ``invalid_policy`` is ``"nan"``.
    """

    depth: npt.NDArray[np.float32]
    valid_mask: npt.NDArray[np.bool_]
    timestamp_us: int
    camera: str
    units: Literal["m"] = "m"
    representation: Literal["ray_distance"] = "ray_distance"


class DepthReader(_AuxiliaryReader):
    """Read individual camera frames from a local Zarr directory or .itar file.

    ``camera_names`` lists available cameras. ``timestamps`` maps each camera to
    sorted int64 microseconds, and ``camera_metadata`` preserves its stored attributes.
    Close the reader after use, or use it as a context manager.
    """

    def __init__(self, path: str | Path) -> None:
        """Open a depth sidecar without loading its images."""
        super().__init__(path, "depth")
        try:
            self.camera_names: list[str] = sorted(self._group.group_keys())
            self.camera_metadata: dict[str, dict[str, object]] = {}
            self.timestamps: dict[str, npt.NDArray[np.int64]] = {}
            for camera in self.camera_names:
                group = self._group[camera]
                self.camera_metadata[camera] = dict(group.attrs)
                self.timestamps[camera] = np.asarray(
                    sorted(int(key) for key in group.array_keys()), dtype=np.int64
                )
        except BaseException:
            self.close()
            raise

    def get_frame(
        self,
        camera: str,
        *,
        frame_index: int | None = None,
        timestamp_us: int | None = None,
        invalid_policy: Literal["preserve", "nan"] = "preserve",
    ) -> DepthFrame:
        """Load one depth image, selecting by index or microsecond timestamp.

        Timestamps select the closest preceding frame and must lie within the
        camera's available range. Omit both selectors for the first frame. ``"nan"`` replaces
        non-positive and non-finite values with NaN; ``"preserve"`` retains them.
        """
        self._ensure_open()
        if invalid_policy not in {"preserve", "nan"}:
            raise ValueError(f"Unknown invalid_policy: {invalid_policy!r}")
        timestamp = _frame_timestamp(
            self.timestamps[camera], frame_index=frame_index, timestamp_us=timestamp_us
        )
        dataset = self._group[camera][str(timestamp)]
        metadata = self.camera_metadata[camera]
        if metadata.get("store_depth_as_png", False):
            max_depth_m = float(metadata["max_depth_m"])
            if not np.isfinite(max_depth_m) or max_depth_m <= 0:
                raise ValueError("PNG depth requires a finite, positive max_depth_m")
            with av.open(io.BytesIO(dataset[()]), format="image2pipe") as container:
                image = next(container.decode(video=0))
                depth = image.to_ndarray(format="gray16le").astype(np.float32)
            depth = depth * max_depth_m / np.iinfo(np.uint16).max
        else:
            depth = np.asarray(dataset, dtype=np.float32)
        valid_mask = np.isfinite(depth) & (depth > 0)
        if invalid_policy == "nan":
            depth[~valid_mask] = np.nan
        return DepthFrame(depth, valid_mask, timestamp, camera)
