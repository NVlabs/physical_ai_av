"""Check remote F4D metadata, label availability and PAI source routing."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import huggingface_hub
import pandas as pd
import pytest

from physical_ai_av import F4DDatasetInterface
from physical_ai_av.f4d import F4D_REPO_ID

PAI_REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
FIRST = "00000000-0000-0000-0000-000000000001"
SECOND = "00000000-0000-0000-0000-000000000002"
UNLABELLED = "00000000-0000-0000-0000-000000000004"


def _label_filename(clip_id: str, feature: str) -> str:
    return f"clips/pai_{clip_id}/pai_{clip_id}.aux.{feature}.zarr.itar"


@pytest.fixture
def remote_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Mock:
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits/test-total.txt").write_text(
        f"pai_{SECOND}\npai_{FIRST}\npai_{UNLABELLED}\n"
    )
    features = pd.DataFrame(
        [
            ["camera_front_wide_120fov", "camera"],
            ["egomotion", "labels"],
            ["sensor_extrinsics", "calibration"],
            ["lidar_intrinsics.offline", "calibration"],
            ["lidar_top", "lidar"],
        ],
        columns=["feature", "directory"],
    ).set_index("feature")
    features[["chunk_path", "clip_files_in_zip"]] = None
    features.to_csv(tmp_path / "features.csv")
    clip_index = pd.DataFrame({"chunk": [10, 11]}, index=pd.Index([FIRST, SECOND], name="clip_id"))
    clip_index.to_parquet(tmp_path / "clip_index.parquet")
    presence = pd.DataFrame(True, index=clip_index.index, columns=features.index)
    presence.loc[SECOND, "camera_front_wide_120fov"] = False
    (tmp_path / "metadata").mkdir()
    presence.to_parquet(tmp_path / "metadata/feature_presence.parquet")
    pd.DataFrame(index=clip_index.index).to_parquet(tmp_path / "metadata/data_collection.parquet")
    label_files = [
        _label_filename(FIRST, "depth"),
        _label_filename(FIRST, "occupancy"),
        _label_filename(SECOND, "depth"),
    ]
    api = Mock()
    api.list_repo_tree.return_value = [SimpleNamespace(path=path) for path in label_files]
    api.file_exists.return_value = True
    api.get_paths_info.return_value = [SimpleNamespace(size=1)]
    api.hf_hub_download.side_effect = lambda filename, **kwargs: str(tmp_path / filename)
    monkeypatch.setattr(huggingface_hub, "HfApi", Mock(return_value=api))
    monkeypatch.setattr(huggingface_hub, "HfFileSystem", Mock())
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", Mock(return_value=None))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    if hasattr(huggingface_hub, "is_offline_mode"):
        monkeypatch.setattr(huggingface_hub, "is_offline_mode", lambda: False)
    return api


def test_f4d_metadata_and_default_sources(remote_metadata: Mock, tmp_path: Path) -> None:
    """Keep the F4D split and availability while routing PAI features independently."""
    dataset = F4DDatasetInterface(
        "f4d-revision",
        cache_dir=tmp_path / "f4d-cache",
        pai_revision="pai-revision",
        pai_cache_dir=tmp_path / "pai-cache",
    )
    assert dataset.repo_id == F4D_REPO_ID
    assert dataset.pai_source.repo_id == PAI_REPO_ID
    assert dataset.feature_sources == {
        feature: dataset.pai_source
        for feature in (
            "camera_front_wide_120fov",
            "egomotion",
            "sensor_extrinsics",
            "lidar_intrinsics.offline",
        )
    }
    assert "lidar_top" not in dataset.features.ALL
    assert dataset.revision == "f4d-revision"
    assert dataset.pai_source.revision == "pai-revision"
    assert dataset.cache_dir == tmp_path / "f4d-cache"
    assert dataset.pai_source.cache_dir == tmp_path / "pai-cache"
    for call in remote_metadata.hf_hub_download.call_args_list:
        expected = dataset if call.kwargs["repo_id"] == F4D_REPO_ID else dataset.pai_source
        assert call.kwargs["revision"] == expected.revision
        assert call.kwargs["cache_dir"] == expected.cache_dir
    assert dataset.clip_index.index.tolist() == [SECOND, FIRST, UNLABELLED]
    assert dataset.feature_presence[["depth", "occupancy"]].to_dict("index") == {
        SECOND: {"depth": True, "occupancy": False},
        FIRST: {"depth": True, "occupancy": True},
        UNLABELLED: {"depth": False, "occupancy": False},
    }
    assert dataset.feature_presence["camera_front_wide_120fov"].tolist() == [False, True, False]


@pytest.mark.parametrize(
    "feature,reader_name", [("depth", "DepthReader"), ("occupancy", "OccupancyReader")]
)
def test_label_decoders_use_downloaded_indexed_tar(
    remote_metadata: Mock,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    feature: str,
    reader_name: str,
) -> None:
    """Dispatch each label to its reader after resolving the downloaded archive."""
    dataset = F4DDatasetInterface("f4d-revision", pai_revision="pai-revision")
    reader = Mock()
    monkeypatch.setattr(f"physical_ai_av.{feature}.{reader_name}", reader)
    with pytest.raises(FileNotFoundError, match="not cached"):
        dataset.get_clip_feature(FIRST, feature)
    assert dataset.get_clip_feature(FIRST, feature, maybe_stream=True) is reader.return_value
    filename = _label_filename(FIRST, feature)
    reader.assert_called_once_with(str(tmp_path / filename))
    assert remote_metadata.hf_hub_download.call_args.kwargs["repo_id"] == F4D_REPO_ID
