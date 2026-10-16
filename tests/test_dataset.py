"""Exercise repository routing and selective extraction with real local archives."""

import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from physical_ai_av.dataset import PhysicalAIAVDatasetInterface

REVISION = "a" * 40


@pytest.fixture
def dataset_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[..., PhysicalAIAVDatasetInterface]:
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)
    monkeypatch.setattr("huggingface_hub.is_offline_mode", lambda: True, raising=False)

    def create(
        repo_id: str = "nvidia/PhysicalAI-Autonomous-Vehicles",
        chunk: int = 1,
        payload: bytes = b"original",
        feature_sources: dict[str, PhysicalAIAVDatasetInterface] | None = None,
    ) -> PhysicalAIAVDatasetInterface:
        cache_dir = tmp_path / repo_id.split("/")[-1]
        root = cache_dir / f"datasets--{repo_id.replace('/', '--')}" / "snapshots" / REVISION
        (root / "metadata").mkdir(parents=True)
        pd.DataFrame(
            [
                (
                    "payload",
                    "labels",
                    "chunk_{chunk_id}.zip",
                    json.dumps({"data": "{clip_id}/data.bin"}),
                ),
                ("direct", "labels", "chunk_{chunk_id}.parquet", None),
            ],
            columns=["feature", "directory", "chunk_path", "clip_files_in_zip"],
        ).to_csv(root / "features.csv", index=False)
        pd.DataFrame({"chunk": [chunk]}, index=["clip"]).to_parquet(root / "clip_index.parquet")
        pd.DataFrame({"payload": [True], "direct": [True]}, index=["clip"]).to_parquet(
            root / "metadata/feature_presence.parquet"
        )
        pd.DataFrame(index=["clip"]).to_parquet(root / "metadata/data_collection.parquet")
        with zipfile.ZipFile(root / f"chunk_{chunk}.zip", "w") as archive:
            archive.writestr("clip/data.bin", payload)
        return PhysicalAIAVDatasetInterface(
            REVISION, repo_id=repo_id, cache_dir=cache_dir, feature_sources=feature_sources
        )

    return create


def test_selected_source_owns_chunk_cache_and_decoder(
    dataset_factory: Callable[..., PhysicalAIAVDatasetInterface],
) -> None:
    """Read and extract the selected source's clip rather than the requester's."""
    source = dataset_factory(repo_id="example/source", chunk=8, payload=b"source")
    dataset = dataset_factory(feature_sources={"payload": source})
    decoded = dataset.get_clip_feature("clip", "payload")
    assert decoded["data"].read() == b"source"
    path = dataset.get_clip_feature_paths("clip", "payload", members="data")["data"]
    assert path.read_bytes() == b"source"
    assert path.is_relative_to(source.cache_dir)


def test_download_groups_features_by_source_chunk(
    dataset_factory: Callable[..., PhysicalAIAVDatasetInterface],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Download each feature from its owning source, including a forwarded source."""
    source = dataset_factory(repo_id="example/source", chunk=8)
    forwarder = dataset_factory(repo_id="example/forwarder", feature_sources={"payload": source})
    forwarder.features.ALL.remove("payload")
    dataset = dataset_factory(feature_sources={"payload": forwarder})
    local_download = Mock()
    source_download = Mock()
    monkeypatch.setattr(dataset, "download_files", local_download)
    monkeypatch.setattr(source, "download_files", source_download)
    dataset.download_clip_features("clip", ["direct", "payload"], dry_run=True)
    local_download.assert_called_once_with(["chunk_1.parquet"], dry_run=True)
    source_download.assert_called_once_with(["chunk_8.zip"], dry_run=True)
