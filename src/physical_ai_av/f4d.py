# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Read Foundation4D labels and their matching Physical AI AV sensor features."""

import hashlib
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

import huggingface_hub.constants
import huggingface_hub.utils as hf_utils
import pandas as pd

from physical_ai_av.dataset import Features, PhysicalAIAVDatasetInterface

if TYPE_CHECKING:
    from physical_ai_av.depth import DepthReader
    from physical_ai_av.occupancy import OccupancyReader

F4D_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles-Foundation4D"
_LABEL_FEATURES = ("depth", "occupancy")


class F4DDatasetInterface(PhysicalAIAVDatasetInterface):
    """Read F4D's ordered split, depth and occupancy, plus matching PAI features.

    Clip IDs use the same UUID form as Physical AI AV. The F4D ``pai_`` prefix
    appears only in stored paths and is removed when reading the split.
    ``feature_presence`` keeps every split member, including clips missing labels.
    """

    def __init__(
        self,
        revision: str | None = None,
        *,
        repo_id: str = F4D_REPO_ID,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_dir: str | Path | None = None,
        confirm_download_threshold_gb: float = 10.0,
        pai_source: PhysicalAIAVDatasetInterface | None = None,
        pai_revision: str | None = None,
        pai_cache_dir: str | Path | None = None,
    ) -> None:
        """Load F4D metadata and its matching PAI sensor source.

        Args:
            revision: F4D repository revision; supply a pinned revision for offline use.
            repo_id: F4D repository to read.
            token: Hugging Face authentication, inherited by the default PAI source.
            cache_dir: Cache for F4D downloads and its file inventory.
            local_dir: Optional destination for F4D downloads.
            confirm_download_threshold_gb: Download confirmation threshold in GiB.
            pai_source: Existing PAI source for sensor features.
            pai_revision: Revision for the default remote PAI source.
            pai_cache_dir: Separate PAI download cache; F4D's local directory is not reused.
        """
        self.pai_source = (
            pai_source
            if pai_source is not None
            else PhysicalAIAVDatasetInterface(
                revision=pai_revision,
                token=token,
                cache_dir=pai_cache_dir,
                confirm_download_threshold_gb=confirm_download_threshold_gb,
            )
        )
        super().__init__(
            revision=revision,
            repo_id=repo_id,
            token=token,
            cache_dir=cache_dir,
            local_dir=local_dir,
            confirm_download_threshold_gb=confirm_download_threshold_gb,
        )

    def _load_metadata(self) -> None:
        split_path = self.download_file("splits/test-total.txt")
        if not isinstance(split_path, str):
            raise FileNotFoundError("F4D split download was skipped")
        clip_ids = [
            _clip_uuid(line) for line in Path(split_path).read_text().splitlines() if line.strip()
        ]
        filenames = self._remote_label_files()
        if len(set(clip_ids)) != len(clip_ids):
            raise ValueError("F4D clip IDs must be unique")
        self.clip_index = pd.DataFrame(index=pd.Index(clip_ids, name="clip_id"))
        self.feature_presence = pd.DataFrame(
            {
                feature: [
                    self._get_clip_feature_filename(clip_id, feature) in filenames
                    for clip_id in clip_ids
                ]
                for feature in _LABEL_FEATURES
            },
            index=self.clip_index.index,
            dtype=bool,
        )
        features_df = pd.DataFrame(
            {"directory": "labels", "chunk_path": None, "clip_files_in_zip": None},
            index=pd.Index(_LABEL_FEATURES, name="feature"),
        )
        self.features = Features(self._add_pai_features(features_df, self.pai_source))

    def _add_pai_features(
        self, features_df: pd.DataFrame, source: PhysicalAIAVDatasetInterface
    ) -> pd.DataFrame:
        source_features = [
            feature
            for feature, directory in source.features.features_df["directory"].items()
            if directory in {"camera", "calibration"} or feature == "egomotion"
        ]
        for feature in source_features:
            self.feature_sources[feature] = source
            present = pd.Series(
                self.clip_index.index.isin(source.clip_index.index), index=self.clip_index.index
            )
            if feature in source.feature_presence:
                present &= source.feature_presence[feature].reindex(
                    self.clip_index.index, fill_value=False
                )
            self.feature_presence[feature] = present
        return pd.concat([features_df, source.features.features_df.loc[source_features]])

    def _remote_label_files(self) -> set[str]:
        cache_key = hashlib.sha256(f"{self.repo_id}@{self.revision}".encode()).hexdigest()
        cache_root = Path(self.cache_dir or huggingface_hub.constants.HF_HUB_CACHE)
        manifest = cache_root / "physical_ai_av" / cache_key / "f4d-files.json"
        if self.is_offline_mode:
            try:
                return set(json.loads(manifest.read_text()))
            except FileNotFoundError as error:
                raise hf_utils.OfflineModeIsEnabled(
                    "F4D file inventory is not cached; initialize this revision online first."
                ) from error
        filenames = {
            entry.path
            for entry in self.api.list_repo_tree(
                path_in_repo="clips", recursive=True, **self.repo_snapshot_info
            )
            if entry.path.endswith((".aux.depth.zarr.itar", ".aux.occupancy.zarr.itar"))
        }
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=manifest.parent) as temporary_dir:
            temporary_manifest = Path(temporary_dir) / manifest.name
            temporary_manifest.write_text(json.dumps(sorted(filenames)))
            temporary_manifest.replace(manifest)
        return filenames

    def _get_clip_feature_filename(self, clip_id: str, feature: str) -> str:
        return f"clips/pai_{clip_id}/pai_{clip_id}.aux.{feature}.zarr.itar"

    def _get_clip_feature_source(self, clip_id: str, feature: str) -> PhysicalAIAVDatasetInterface:
        source = super()._get_clip_feature_source(clip_id, feature)
        if feature in _LABEL_FEATURES and not self.feature_presence.at[clip_id, feature]:
            raise KeyError(f"{feature!r} is not available for clip {clip_id!r}.")
        return source

    def _decode_clip_feature(
        self, clip_id: str, feature: str, filename: str, maybe_stream: bool = False
    ) -> "DepthReader | OccupancyReader":
        from physical_ai_av.depth import DepthReader
        from physical_ai_av.occupancy import OccupancyReader

        path = str(
            self.get_clip_feature_paths(clip_id, feature, maybe_stream=maybe_stream)[feature]
        )
        return DepthReader(path) if feature == "depth" else OccupancyReader(path)

    def _download_clip_features(
        self, clip_ids: list[str], features: list[str], **kwargs: object
    ) -> None:
        filenames = [
            self._get_clip_feature_filename(clip_id, feature)
            for clip_id in clip_ids
            for feature in features
            if self.feature_presence.at[clip_id, feature]
        ]
        self.download_files(filenames, **kwargs)


def _clip_uuid(clip_id: str) -> str:
    return str(UUID(clip_id.strip().removeprefix("pai_")))
