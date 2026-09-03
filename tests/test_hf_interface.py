import huggingface_hub
import pytest
from huggingface_hub.hf_api import RepoFile

from physical_ai_av.utils.hf_interface import HfRepoInterface


def _repo_with_local_dir(local_dir):
    return HfRepoInterface(
        repo_id="org/repo",
        repo_type="dataset",
        revision="abcdef",
        local_dir=local_dir,
    )


def _repo_file(path, size):
    try:
        return RepoFile(path=path, size=size, oid="")
    except (KeyError, TypeError):
        return RepoFile(path=path, size=size, blob_id="")


def test_open_file_prefers_materialized_local_dir_file(tmp_path, monkeypatch):
    local_file = tmp_path / "features" / "chunk_0.zip"
    local_file.parent.mkdir()
    local_file.write_bytes(b"local")

    cache_file = tmp_path / "cache_file"
    cache_file.write_bytes(b"cache")
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda *args, **kwargs: str(cache_file),
    )

    repo = _repo_with_local_dir(tmp_path)
    with repo.open_file("features/chunk_0.zip") as f:
        assert f.read() == b"local"


def test_download_file_returns_materialized_local_dir_file_without_remote_lookup(tmp_path):
    local_file = tmp_path / "metadata" / "sensor_presence.parquet"
    local_file.parent.mkdir()
    local_file.write_bytes(b"local")

    class FailingApi:
        def get_paths_info(self, *args, **kwargs):
            raise AssertionError("download_file should not query remote metadata")

        def hf_hub_download(self, *args, **kwargs):
            raise AssertionError("download_file should not download materialized local files")

    repo = _repo_with_local_dir(tmp_path)
    repo.api = FailingApi()

    assert repo.download_file("metadata/sensor_presence.parquet") == str(local_file)


def test_download_file_resolves_materialized_subfolder_file(tmp_path):
    local_file = tmp_path / "metadata" / "sensor_presence.parquet"
    local_file.parent.mkdir()
    local_file.write_bytes(b"local")

    repo = _repo_with_local_dir(tmp_path)

    assert repo.download_file("sensor_presence.parquet", subfolder="metadata") == str(local_file)


def test_download_files_returns_materialized_local_dir_files_without_remote_lookup(tmp_path):
    local_files = [
        tmp_path / "features" / "chunk_0.zip",
        tmp_path / "metadata" / "sensor_presence.parquet",
    ]
    for local_file in local_files:
        local_file.parent.mkdir(exist_ok=True)
        local_file.write_bytes(b"local")

    class FailingApi:
        def get_paths_info(self, *args, **kwargs):
            raise AssertionError("download_files should not query remote metadata")

        def hf_hub_download(self, *args, **kwargs):
            raise AssertionError("download_files should not download materialized local files")

    repo = _repo_with_local_dir(tmp_path)
    repo.api = FailingApi()

    assert repo.download_files(
        ["features/chunk_0.zip", "metadata/sensor_presence.parquet"]
    ) == [str(local_file) for local_file in local_files]


def test_download_files_rejects_subfolder_even_when_file_is_materialized(tmp_path):
    local_file = tmp_path / "features" / "chunk_0.zip"
    local_file.parent.mkdir()
    local_file.write_bytes(b"local")

    repo = _repo_with_local_dir(tmp_path)

    with pytest.raises(ValueError, match="subfolder"):
        repo.download_files(["features/chunk_0.zip"], subfolder="features")


def test_download_files_merges_materialized_and_downloaded_files(tmp_path, monkeypatch):
    local_file = tmp_path / "features" / "chunk_0.zip"
    local_file.parent.mkdir()
    local_file.write_bytes(b"local")
    downloaded_file = tmp_path / "downloads" / "remote_feature.zip"

    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda *args, **kwargs: None,
    )

    class Api:
        def get_paths_info(self, paths, *args, **kwargs):
            assert paths == ["features/remote_feature.zip"]
            return [_repo_file(path=paths[0], size=1)]

        def hf_hub_download(self, filename, *args, **kwargs):
            assert filename == "features/remote_feature.zip"
            return str(downloaded_file)

    repo = _repo_with_local_dir(tmp_path)
    repo.api = Api()

    assert repo.download_files(
        ["features/chunk_0.zip", "features/remote_feature.zip"], max_workers=1
    ) == [str(local_file), str(downloaded_file)]
