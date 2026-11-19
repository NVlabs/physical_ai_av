# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
import abc
import dataclasses

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclasses.dataclass
class CameraModel(abc.ABC):
    width: int
    height: int

    def is_out_of_bounds(self, pixel: npt.NDArray[np.number]) -> npt.NDArray[np.bool_]:
        """Checks if pixels are out of bounds.

        Args:
            pixel: Pixel coordinates. [..., 2]

        Returns:
            out_of_bounds: Whether pixels are out of bounds. [...]
                True if pixel is out of bounds, False otherwise.
        """
        return (
            (pixel[..., 0] < 0)
            | (pixel[..., 0] >= self.width)
            | (pixel[..., 1] < 0)
            | (pixel[..., 1] >= self.height)
        )

    @abc.abstractmethod
    def ray2pixel(self, ray: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Converts rays relative to camera frame to pixel coordinates.

        Args:
            ray: Ray vector in camera frame. [..., 3]
                Camera frame is right-handed, with Z-axis pointing out of the camera.
                X-axis is pointing right, and Y-axis is pointing down.

        Returns:
            pixel: Pixel coordinates. [..., 2]
                Pixel coordinates are in the image plane, with the origin at the top-left corner.
                X-axis is pointing right, and Y-axis is pointing down.
        """
        pass

    @abc.abstractmethod
    def pixel2ray(self, pixel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Converts pixel coordinates to rays relative to camera frame.

        Args:
            pixel: Pixel coordinates. [..., 2]
                Pixel coordinates are in the image plane, with the origin at the top-left corner.
                X-axis is pointing right, and Y-axis is pointing down.

        Returns:
            ray: Ray vector in camera frame. [..., 3]
                Camera frame is right-handed, with Z-axis pointing out of the camera.
                X-axis is pointing right, and Y-axis is pointing down.

            Rays are normalized to have unit length.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_camera_row(cls, camera_row: pd.Series) -> "CameraModel":
        """Loads Camera Model Parameters from CALIBRATION.INTRINSICS feature.

        Args:
            camera_row (pd.Series): Camera row from CALIBRATION.INTRINSICS feature.

        Returns:
            CameraModel.
        """
        pass


@dataclasses.dataclass
class PinholeCameraModel(CameraModel):
    camera_matrix: np.ndarray

    @classmethod
    def from_camera_row(cls, camera_row: pd.Series) -> "PinholeCameraModel":
        """Loads Camera Model Parameters from CALIBRATION.INTRINSICS feature.

        Args:
            camera_row (pd.Series): Camera row from CALIBRATION.INTRINSICS feature.

        Returns:
            PinholeCameraModel.
        """
        width = int(camera_row["width"])
        height = int(camera_row["height"])
        f, cx, cy = camera_row["fw_poly_1"], camera_row["cx"], camera_row["cy"]
        return cls(
            width,
            height,
            np.array(
                [
                    [f, 0, cx],
                    [0, f, cy],
                    [0, 0, 1],
                ]
            ),
        )

    def ray2pixel(self, ray: np.ndarray) -> np.ndarray:
        pixel = np.einsum("ij,...j->...i", self.camera_matrix, ray)
        return pixel[..., :2] / pixel[..., 2:]

    def pixel2ray(self, pixel: np.ndarray) -> np.ndarray:
        pixel = np.concatenate([pixel, np.ones_like(pixel[..., :1])], axis=-1)
        ray = np.einsum("ij,...j->...i", np.linalg.inv(self.camera_matrix), pixel)
        return ray / np.linalg.norm(ray, ord=2, axis=-1, keepdims=True)


@dataclasses.dataclass
class FThetaCameraModel(CameraModel):
    principal_point: np.ndarray
    th2r: np.polynomial.Polynomial
    r2th: np.polynomial.Polynomial

    @classmethod
    def from_camera_row(
        cls, camera_row: pd.Series, polynomial_degree: int = 4
    ) -> "FThetaCameraModel":
        """Loads Camera Model Parameters from CALIBRATION.INTRINSICS feature.

        Args:
            camera_row (pd.Series): Camera row from CALIBRATION.INTRINSICS feature.
            polynomial_degree (int): Degree of the polynomial.

        Returns:
            FThetaCameraModel.
        """
        width = int(camera_row["width"])
        height = int(camera_row["height"])
        return cls(
            width=width,
            height=height,
            principal_point=np.array([camera_row["cx"], camera_row["cy"]]),
            th2r=np.polynomial.Polynomial(
                [camera_row[f"fw_poly_{i}"] for i in range(polynomial_degree + 1)], symbol="th"
            ),
            r2th=np.polynomial.Polynomial(
                [camera_row[f"bw_poly_{i}"] for i in range(polynomial_degree + 1)], symbol="r"
            ),
        )

    def ray2pixel(self, ray: np.ndarray) -> np.ndarray:
        th = np.arccos(ray[..., 2:] / np.linalg.norm(ray, axis=-1, keepdims=True))
        return self.principal_point + self.th2r(th) * ray[..., :2] / np.linalg.norm(
            ray[..., :2], axis=-1, keepdims=True
        )

    def pixel2ray(self, pixel: np.ndarray) -> np.ndarray:
        p = pixel - self.principal_point
        r = np.linalg.norm(p, axis=-1)
        th = self.r2th(r)
        c, s = np.cos(th), np.sin(th)
        return np.stack([s * p[..., 0] / r, s * p[..., 1] / r, c], -1)
