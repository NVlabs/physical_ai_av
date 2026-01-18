# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
import dataclasses
import json
import io
import logging
import pathlib
import types
import zipfile
from typing import Any, Iterable

import pandas as pd
from physical_ai_av import egomotion, video
from physical_ai_av.utils import hf_interface

import importlib.util

logger = logging.getLogger(__name__)


class PhysicalAIAVDatasetInterface(hf_interface.HfRepoInterface):
    """Interface for interacting with the PhysicalAI-Autonomous-Vehicles dataset on Hugging Face.

    See also the parent class `hf_interface.HfRepoInterface` for additional attributes.

    Attributes:
        revision (`str`): A Git revision id, which can be a branch name, a tag, or a commit hash
            (if not supplied at initialization, the latest commit hash on `main` will be used).
        token (`str | bool | None`): A valid user access token (string). Defaults to the locally
            saved token, which is the recommended method for authentication (see
            https://huggingface.co/docs/huggingface_hub/quick-start#authentication).
            To disable authentication, pass `False`.
        cache_dir (`str | pathlib.Path | None`): Path to the dir where cached files are stored.
        local_dir (`str | pathlib.Path | None`): If provided, downloaded files will be placed under
            this directory.
        confirm_download_threshold_gb (`float`): The threshold (in GB) of additional (uncached) file
            size beyond which the user is prompted for confirmation before downloading. Set to
            `float("inf")` to disable confirmation.
        features (`Features`): A representation of dataset features amenable to `.`-autocompletion.
        clip_index (`pd.DataFrame`): A clip index mapping `clip_id`s to `chunk` indices.
        sensor_presence (`pd.DataFrame`): A table mapping `clip_id`s to available sensors (notably,
            includes the radar config & radar sensor models for each clip).
        chunk_sensor_presence (`pd.DataFrame`): A table of sensor presence aggregated by chunk; used
            to determine which per-chunk packed files should exist in the dataset.
    """

    def __init__(
        self,
        revision: str | None = None,
        *,
        token: str | bool | None = None,
        cache_dir: str | pathlib.Path | None = None,
        local_dir: str | pathlib.Path | None = None,
        confirm_download_threshold_gb: float = 10.0,
    ) -> None:
        super().__init__(
            repo_id="nvidia/PhysicalAI-Autonomous-Vehicles",
            repo_type="dataset",
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_dir=local_dir,
            confirm_download_threshold_gb=confirm_download_threshold_gb,
        )
        features_df = pd.read_csv(self.download_file("features.csv"), index_col="feature")
        features_df["clip_files_in_zip"] = features_df["clip_files_in_zip"].map(
            json.loads, na_action="ignore"
        )
        self.features = Features(features_df)

        self.clip_index = pd.read_parquet(self.download_file("clip_index.parquet"))
        self.sensor_presence = pd.read_parquet(
            self.download_file("metadata/sensor_presence.parquet")
        )
        self.chunk_sensor_presence = (
            pd.concat(
                [self.clip_index[["chunk"]], self.sensor_presence.select_dtypes(include=bool)],
                axis=1,
            )
            .groupby("chunk")
            .any()
        )

    def download_metadata(self) -> None:
        """Downloads dataset metadata, e.g., for the purpose of clip/chunk selection."""
        self.metadata = {
            pathlib.Path(f).stem: pd.read_parquet(f) for f in self.download_repo_tree("metadata/")
        }

    def get_clip_chunk(self, clip_id: str) -> int:
        """Returns the chunk index for `clip_id`."""
        return self.clip_index.at[clip_id, "chunk"]

    def download_clip_features(
        self, clip_id: str | Iterable[str], features: str | Iterable[str] | None = None, **kwargs
    ) -> None:
        """Downloads features for specified clip(s); see `download_files` for more kwargs."""
        if isinstance(clip_id, str):
            clip_id = [clip_id]
        self.download_chunk_features(set(self.clip_index.loc[clip_id, "chunk"]), features, **kwargs)

    def download_chunk_features(
        self, chunk_id: int | Iterable[int], features: str | Iterable[str] | None = None, **kwargs
    ) -> None:
        """Downloads features for specified chunk(s); see `download_files` for more kwargs."""
        if features is None:
            features = self.features.ALL
        if isinstance(chunk_id, int):
            chunk_id = [chunk_id]
        if isinstance(features, str):
            features = [features]
        chunk_id = set(chunk_id)
        features = set(features)
        files_to_download = []
        for chunk_id in chunk_id:
            for feature in features:
                if (
                    feature not in self.chunk_sensor_presence.columns
                    or self.chunk_sensor_presence.at[chunk_id, feature]
                ):
                    files_to_download.append(
                        self.features.get_chunk_feature_filename(chunk_id, feature)
                    )
                else:
                    logger.debug(
                        f"Skipping {feature} for chunk {chunk_id} because it does not exist."
                    )
        self.download_files(files_to_download, **kwargs)

    def get_clip_feature(self, clip_id: str, feature: str, maybe_stream: bool = False) -> Any:
        chunk_filename = self.features.get_chunk_feature_filename(
            self.get_clip_chunk(clip_id), feature
        )
        with self.open_file(chunk_filename, maybe_stream=maybe_stream) as f:
            if chunk_filename.endswith(".parquet"):
                return pd.read_parquet(f).loc[clip_id]
            elif chunk_filename.endswith(".zip"):
                clip_files_in_zip = self.features.get_clip_files_in_zip(clip_id, feature)
                with zipfile.ZipFile(f, "r") as zf:
                    if feature == "egomotion":
                        egomotion_df = pd.read_parquet(
                            io.BytesIO(zf.read(clip_files_in_zip["egomotion"]))
                        )
                        return egomotion.EgomotionState.from_egomotion_df(
                            egomotion_df
                        ).create_interpolator(egomotion_df["timestamp"].to_numpy())
                    elif feature.startswith("camera"):
                        video_data = io.BytesIO(zf.read(clip_files_in_zip["video"]))
                        timestamps = pd.read_parquet(
                                io.BytesIO(zf.read(clip_files_in_zip["frame_timestamps"]))
                            )["timestamp"].to_numpy()

                        
                        return video.SeekVideoReader(
                                video_data=video_data,
                                timestamps=timestamps,
                            )

                    else:
                        logger.warning(
                            f"Feature-specific data reader for {feature=} not implemented yet."
                        )
                        return {
                            k: pd.read_parquet(io.BytesIO(zf.read(v)))
                            if v.endswith(".parquet")
                            else io.BytesIO(zf.read(v))
                            for k, v in self.features.get_clip_files_in_zip(
                                clip_id, feature
                            ).items()
                        }
            else:
                raise ValueError(f"Unexpected file extension: {chunk_filename=}.")


class Features:
    """Class for representing dataset features and info on their packed format on Hugging Face."""

    def __init__(self, features_df: pd.DataFrame) -> None:
        self.features_df = features_df

        # Create feature aliases amenable to `.`-autocompletion, e.g., for individual features,
        # `features.CAMERA.CAMERA_FRONT_WIDE_120FOV` or `features.LABELS.EGOMOTION`, and for all
        # features in a directory, `features.CAMERA.ALL` or `features.LABELS.ALL`.
        self.ALL = set()
        for directory, directory_features in self.features_df.groupby("directory"):
            setattr(
                self,
                directory.upper(),
                types.SimpleNamespace(
                    **{feature.upper(): feature for feature in directory_features.index},
                    ALL=set(directory_features.index),
                ),
            )
            self.ALL.update(getattr(self, directory.upper()).ALL)

    def get_chunk_feature_filename(self, chunk_id: int, feature: str):
        """Returns the chunk feature filename within the dataset repo."""
        return self.features_df.at[feature, "chunk_path"].format(chunk_id=chunk_id)

    def get_clip_files_in_zip(self, clip_id: str, feature: str) -> list[str]:
        """Returns the files within a chunk feature zip corresponding to `clip_id`."""
        templates = self.features_df.at[feature, "clip_files_in_zip"]
        if not isinstance(templates, dict):
            raise ValueError(f"{feature=} is not chunked as zip files.")
        return {k: v.format(clip_id=clip_id) for k, v in templates.items()}
