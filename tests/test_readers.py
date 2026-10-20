# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Check depth and occupancy decoding from local auxiliary archives."""

import tarfile
from pathlib import Path

import numpy as np
import zarr

from physical_ai_av.depth import DepthReader
from physical_ai_av.occupancy import OccupancyReader


def _archive(directory: Path) -> Path:
    path = directory.with_suffix(".itar")
    with tarfile.open(path, "w") as archive:
        archive.add(directory, arcname=".")
    return path


def test_depth(tmp_path: Path) -> None:
    """Read metric depth and mark invalid values."""
    directory = tmp_path / "depth.zarr"
    root = zarr.open_group(str(directory), mode="w")
    root.attrs["aux_root_group_name"] = "aux"
    expected = np.array([[1.25, 0, np.nan]], dtype=np.float32)
    root.create_dataset("aux/depth/front/10", data=expected.astype(np.float16))
    with DepthReader(_archive(directory)) as reader:
        frame = reader.get_frame("front")
    np.testing.assert_array_equal(frame.depth, expected)
    np.testing.assert_array_equal(frame.valid_mask, [[True, False, False]])
    assert frame.depth.dtype == np.float32
    assert (frame.timestamp_us, frame.units) == (10, "m")


def test_occupancy(tmp_path: Path) -> None:
    """Read sparse voxel fields and grid metadata."""
    directory = tmp_path / "occupancy.zarr"
    root = zarr.open_group(str(directory), mode="w")
    root.attrs["aux_root_group_name"] = "aux"
    stored = root.create_group("aux/occupancy/lidar/10")
    values = {
        "voxel_coords": [[1, 2, 3]],
        "occupancy_probs": [0.75],
        "semantic_labels": [4],
        "velocity": [[1.0, 2.0, 3.0]],
        "grid_shape": [8, 9, 10],
        "grid_origin": [-1.0, -2.0, -3.0],
        "voxel_size": [0.5],
    }
    for name, value in values.items():
        stored.create_dataset(name, data=np.asarray(value))
    with OccupancyReader(_archive(directory)) as reader:
        frame = reader.get_frame()
    for name in ("voxel_coords", "occupancy_probs", "semantic_labels", "grid_origin"):
        np.testing.assert_array_equal(getattr(frame, name), values[name])
    np.testing.assert_array_equal(frame.velocity_vectors, values["velocity"])
    assert (frame.timestamp_us, frame.grid_shape, frame.voxel_size) == (10, (8, 9, 10), 0.5)
    assert (frame.units, frame.velocity_units) == ("m", "m/s")
