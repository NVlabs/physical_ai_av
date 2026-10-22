# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
import hashlib
import io
import json
import logging
import pathlib
import re
import shutil
import tempfile
import types
import uuid
import zipfile
from typing import Any, Iterable, Mapping

import huggingface_hub
import huggingface_hub.utils as hf_utils
import pandas as pd

from physical_ai_av import calibration, egomotion, video
from physical_ai_av.utils import hf_interface

logger = logging.getLogger(__name__)


class PhysicalAIAVDatasetInterface(hf_interface.HfRepoInterface):
    """Interface for interacting with the PhysicalAI-Autonomous-Vehicles dataset on Hugging Face.

    See also the parent class `hf_interface.HfRepoInterface` for additional attributes.

    Attributes:
        repo_id (`str`): Dataset repository; defaults to PhysicalAI-Autonomous-Vehicles.
        feature_sources (`Mapping`): Optional feature-to-dataset mapping. Each source supplies
            its own clip index, cache settings, and decoder for the mapped features.
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
        feature_presence (`pd.DataFrame`): A table mapping `clip_id`s to available features.
        chunk_feature_presence (`pd.DataFrame`): A table of feature presence aggregated by chunk;
            used to determine which per-chunk packed files should exist in the dataset.
    """

    def __init__(
        self,
        revision: str | None = None,
        *,
        repo_id: str = "nvidia/PhysicalAI-Autonomous-Vehicles",
        feature_sources: Mapping[str, "PhysicalAIAVDatasetInterface"] | None = None,
        token: str | bool | None = None,
        cache_dir: str | pathlib.Path | None = None,
        local_dir: str | pathlib.Path | None = None,
        confirm_download_threshold_gb: float = 10.0,
        split_name: str = "default",
    ) -> None:
        self.split_name = split_name
        super().__init__(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_dir=local_dir,
            confirm_download_threshold_gb=confirm_download_threshold_gb,
        )
        self.feature_sources = dict(feature_sources or {})
        self._load_metadata()

    def _load_metadata(self) -> None:
        features_df = pd.read_csv(self.download_file("features.csv"), index_col="feature")
        features_df["clip_files_in_zip"] = features_df["clip_files_in_zip"].map(
            json.loads, na_action="ignore"
        )
        self.features = Features(features_df)

        self.clip_index = pd.read_parquet(self.download_file("clip_index.parquet"))
        if self.split_name != "default" or self._has_file("splits/default.parquet"):
            self.clip_index["split"] = self._load_split_labels(self.split_name)
        if (
            self.is_offline_mode
            and not self.is_file_cached("metadata/feature_presence.parquet")
            and not self.is_file_cached("metadata/sensor_presence.parquet")
        ):
            raise hf_utils.OfflineModeIsEnabled(
                "Offline mode is enabled and neither `metadata/feature_presence.parquet` (current) "
                "nor `metadata/sensor_presence.parquet` (legacy, pre-26.03) are cached."
            )
        if self.is_file_cached("metadata/feature_presence.parquet") or self.api.file_exists(
            filename="metadata/feature_presence.parquet", **self.repo_snapshot_info
        ):
            self.feature_presence = pd.read_parquet(
                self.download_file("metadata/feature_presence.parquet")
            )
        else:
            # Fallback to older `sensor_presence.parquet` in dataset revisions prior to 26.03.
            self.sensor_presence = pd.read_parquet(
                self.download_file("metadata/sensor_presence.parquet")
            )
            self.feature_presence = self.sensor_presence.select_dtypes(include=bool)

        self.data_collection = pd.read_parquet(
            self.download_file("metadata/data_collection.parquet")
        )
        self.chunk_feature_presence = (
            pd.concat(
                [self.clip_index[["chunk"]], self.feature_presence],
                axis=1,
            )
            .groupby("chunk")
            .any()
        )

    def _has_file(self, filename: str) -> bool:
        """Returns whether `filename` is cached locally or, when online, exists in the repository."""
        return self.is_file_cached(filename) or (
            not self.is_offline_mode
            and self.api.file_exists(filename=filename, **self.repo_snapshot_info)
        )

    def _load_split_labels(self, split_name: str) -> pd.Series:
        """Returns the `split` label of every clip in `clip_index` from `splits/<split_name>.parquet`."""
        labels = pd.read_parquet(self.download_file(f"splits/{split_name}.parquet"))["split"]
        unknown = labels.index.difference(self.clip_index.index)
        if len(unknown):
            raise ValueError(
                f"splits/{split_name}.parquet labels {len(unknown)} clips absent from "
                f"clip_index.parquet, e.g. {unknown[:3].tolist()}."
            )
        return labels.reindex(self.clip_index.index)

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
        clip_ids = [clip_id] if isinstance(clip_id, str) else list(clip_id)
        self.clip_index.loc[clip_ids]
        if features is None:
            features = self.features.ALL | self.feature_sources.keys()
        features = [features] if isinstance(features, str) else list(features)
        source_features: dict[PhysicalAIAVDatasetInterface, list[str]] = {}
        for feature in features:
            source = self._get_feature_source(feature)
            source.clip_index.loc[clip_ids]
            if feature not in source.features.ALL | source.feature_sources.keys():
                raise KeyError(f"Unknown feature {feature!r} in {source.repo_id!r}.")
            source_features.setdefault(source, []).append(feature)
        for source, grouped_features in source_features.items():
            if source is self:
                self._download_clip_features(clip_ids, grouped_features, **kwargs)
            else:
                source.download_clip_features(clip_ids, grouped_features, **kwargs)

    def _download_clip_features(self, clip_ids: list[str], features: list[str], **kwargs) -> None:
        self.download_chunk_features(
            set(self.clip_index.loc[clip_ids, "chunk"]), features, **kwargs
        )

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
                    feature not in self.chunk_feature_presence.columns
                    or self.chunk_feature_presence.at[chunk_id, feature]
                ):
                    files_to_download.append(
                        self.features.get_chunk_feature_filename(chunk_id, feature)
                    )
                else:
                    logger.debug(
                        f"Skipping {feature} for chunk {chunk_id} because it does not exist."
                    )
        self.download_files(files_to_download, **kwargs)

    def _get_feature_source(self, feature: str) -> "PhysicalAIAVDatasetInterface":
        return self.feature_sources.get(feature, self)

    def _get_clip_feature_source(
        self, clip_id: str, feature: str
    ) -> "PhysicalAIAVDatasetInterface":
        if clip_id not in self.clip_index.index:
            raise KeyError(f"Unknown clip {clip_id!r} in {self.repo_id!r}.")
        source = self._get_feature_source(feature)
        if source is not self:
            return source._get_clip_feature_source(clip_id, feature)
        if feature not in source.features.ALL:
            raise KeyError(f"Unknown feature {feature!r} in {source.repo_id!r}.")
        return source

    def _get_clip_feature_filename(self, clip_id: str, feature: str) -> str:
        return self.features.get_chunk_feature_filename(self.get_clip_chunk(clip_id), feature)

    def get_clip_feature(self, clip_id: str, feature: str, maybe_stream: bool = False) -> Any:
        """Read a clip feature using its source's metadata, cache, and decoder."""
        source = self._get_clip_feature_source(clip_id, feature)
        return source._decode_clip_feature(
            clip_id, feature, source._get_clip_feature_filename(clip_id, feature), maybe_stream
        )

    def get_clip_feature_paths(
        self,
        clip_id: str,
        feature: str,
        *,
        members: str | Iterable[str] | None = None,
        maybe_stream: bool = False,
    ) -> dict[str, pathlib.Path]:
        """Return cached feature files, extracting only selected ZIP member aliases.

        Args:
            clip_id: The clip to read.
            feature: The feature to resolve through its configured source.
            members: ZIP member aliases, such as ``video`` or ``frame_timestamps``.
                Defaults to all members belonging to the clip.
            maybe_stream: Allow remote reads for uncached files. Direct files are downloaded.

        Returns:
            Member aliases mapped to extracted files, or the feature mapped to its direct file.
        """
        source = self._get_clip_feature_source(clip_id, feature)
        filename = source._get_clip_feature_filename(clip_id, feature)
        if filename.endswith(".zip"):
            return source._extract_clip_feature_paths(
                clip_id, feature, filename, members, maybe_stream
            )
        if members is not None:
            raise ValueError("Member aliases are supported only for ZIP features.")
        filepath = source._get_cached_path(filename)
        if filepath is None and maybe_stream:
            filepath = source.download_file(filename)
        if filepath is None:
            raise FileNotFoundError(
                f"{filename!r} is not cached; set maybe_stream=True to fetch it."
            )
        return {feature: pathlib.Path(filepath)}

    def _extract_clip_feature_paths(
        self,
        clip_id: str,
        feature: str,
        filename: str,
        members: str | Iterable[str] | None,
        maybe_stream: bool,
    ) -> dict[str, pathlib.Path]:
        member_files = self.features.get_clip_files_in_zip(clip_id, feature)
        aliases = (
            member_files
            if members is None
            else ([members] if isinstance(members, str) else members)
        )
        selected = {alias: member_files[alias] for alias in aliases}
        filepath = self._get_cached_path(filename)
        if re.fullmatch(r"[0-9a-fA-F]{40}", self.revision):
            # Keep immutable members reusable when the full archive is downloaded or evicted.
            archive_identity = self.revision
        elif filepath is not None:
            archive_path = pathlib.Path(filepath).resolve()
            archive_stat = archive_path.stat()
            archive_identity = (
                f"{archive_path}:{archive_stat.st_size}:"
                f"{archive_stat.st_mtime_ns}:{archive_stat.st_ctime_ns}"
            )
        else:
            archive_identity = uuid.uuid4().hex
        namespace = hashlib.sha256(
            f"{self.repo_type}/{self.repo_id}@{self.revision}/{filename}/{archive_identity}".encode()
        ).hexdigest()
        root = (
            pathlib.Path(self.cache_dir or huggingface_hub.constants.HF_HUB_CACHE)
            / "subsetcache"
            / namespace
        )
        paths = {}
        for alias, member in selected.items():
            member_path = pathlib.PurePosixPath(member)
            if (
                member_path.is_absolute()
                or ".." in member_path.parts
                or "\\" in member
                or ":" in member
            ):
                raise ValueError(f"Unsafe ZIP member path: {member!r}.")
            target = root / member_path
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"ZIP member escapes the cache directory: {member!r}.")
            paths[alias] = target
        missing = {alias: path for alias, path in paths.items() if not path.is_file()}
        if missing:
            with self.open_file(filename, maybe_stream=maybe_stream) as archive_file:
                with zipfile.ZipFile(archive_file) as archive:
                    for alias, target in missing.items():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = None
                        try:
                            with archive.open(selected[alias]) as source:
                                with tempfile.NamedTemporaryFile(
                                    dir=target.parent, delete=False
                                ) as output:
                                    temporary = pathlib.Path(output.name)
                                    shutil.copyfileobj(source, output)
                            temporary.replace(target)
                        finally:
                            if temporary is not None:
                                temporary.unlink(missing_ok=True)
        return paths

    def _decode_clip_feature(
        self, clip_id: str, feature: str, chunk_filename: str, maybe_stream: bool = False
    ) -> Any:
        with self.open_file(chunk_filename, maybe_stream=maybe_stream) as f:
            if chunk_filename.endswith(".parquet"):
                feature_df = pd.read_parquet(f).loc[clip_id]
                if feature == "sensor_extrinsics":
                    return calibration.SensorExtrinsics.from_extrinsics_df(feature_df)
                elif feature == "camera_intrinsics":
                    return calibration.CameraIntrinsics.from_intrinsics_df(feature_df)
                elif feature == "vehicle_dimensions":
                    return calibration.VehicleDimensions.from_dimensions_df(feature_df)
                else:
                    logger.warning(
                        f"Feature-specific data reader for {feature=} not implemented yet."
                    )
                    return feature_df
            elif chunk_filename.endswith(".zip"):
                clip_files_in_zip = self.features.get_clip_files_in_zip(clip_id, feature)
                with zipfile.ZipFile(f, "r") as zf:
                    if feature == "egomotion":
                        egomotion_df = pd.read_parquet(
                            io.BytesIO(zf.read(clip_files_in_zip["egomotion"]))
                        )
                        return egomotion.EgomotionState.from_egomotion_df(
                            egomotion_df
                        ).create_interpolator(egomotion_df["timestamp"].to_numpy(copy=True))
                    elif feature.startswith("camera"):
                        return video.SeekVideoReader(
                            video_data=io.BytesIO(zf.read(clip_files_in_zip["video"])),
                            timestamps=pd.read_parquet(
                                io.BytesIO(zf.read(clip_files_in_zip["frame_timestamps"]))
                            )["timestamp"].to_numpy(copy=True),
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
                    **{
                        feature.upper().replace(".", "_"): feature
                        for feature in directory_features.index
                    },
                    ALL=set(directory_features.index),
                ),
            )
            self.ALL.update(getattr(self, directory.upper()).ALL)

    def get_chunk_feature_filename(self, chunk_id: int, feature: str):
        """Returns the chunk feature filename within the dataset repo."""
        return self.features_df.at[feature, "chunk_path"].format(chunk_id=chunk_id)

    def get_clip_files_in_zip(self, clip_id: str, feature: str) -> dict[str, str]:
        """Returns the files within a chunk feature zip corresponding to `clip_id`."""
        templates = self.features_df.at[feature, "clip_files_in_zip"]
        if not isinstance(templates, dict):
            raise ValueError(f"{feature=} is not chunked as zip files.")
        return {k: v.format(clip_id=clip_id) for k, v in templates.items()}
