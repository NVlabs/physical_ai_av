# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Check the pinhole camera model against the standard projection."""

import numpy as np
import pandas as pd

from physical_ai_av.utils.camera_models import PinholeCameraModel


def _camera() -> PinholeCameraModel:
    row = pd.Series({"width": 1920, "height": 1080, "fw_poly_1": 1000.0, "cx": 960.0, "cy": 540.0})
    return PinholeCameraModel.from_camera_row(row)


def test_pinhole_ray2pixel() -> None:
    """A ray (x, y, z) projects to (f * x / z + cx, f * y / z + cy)."""
    rays = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0], [-1.0, 0.5, 4.0]])
    expected = np.array([[960.0, 540.0], [1460.0, 540.0], [710.0, 665.0]])
    np.testing.assert_allclose(_camera().ray2pixel(rays), expected)


def test_pinhole_pixel2ray_round_trip() -> None:
    """pixel2ray takes [..., 2] pixels and returns unit rays that project back."""
    camera = _camera()
    pixels = np.array([[960.0, 540.0], [1460.0, 540.0], [710.0, 665.0]])
    rays = camera.pixel2ray(pixels)
    assert rays.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(rays, axis=-1), 1.0)
    np.testing.assert_allclose(rays[1], np.array([1.0, 0.0, 2.0]) / np.sqrt(5.0))
    np.testing.assert_allclose(camera.ray2pixel(rays), pixels)
