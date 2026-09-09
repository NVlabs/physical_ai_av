import logging

import huggingface_hub
import pytest

from physical_ai_av.utils.hf_interface import HfRepoInterface


def _repo_with_local_dir(local_dir, revision="abcdef"):
    return HfRepoInterface(
        repo_id="org/repo",
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
    )


def _set_offline_mode(monkeypatch, is_offline):
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", is_offline)
    monkeypatch.setattr(huggingface_hub, "is_offline_mode", lambda: is_offline, raising=False)


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


def test_download_files_rejects_subfolder(tmp_path):
    repo = _repo_with_local_dir(tmp_path)

    with pytest.raises(ValueError, match="subfolder"):
        repo.download_files(["features/chunk_0.zip"], subfolder="features")


def test_download_files_requires_online_mode_for_materialized_local_dir_files(
    tmp_path, monkeypatch
):
    _set_offline_mode(monkeypatch, True)
    local_file = tmp_path / "features" / "chunk_0.zip"
    local_file.parent.mkdir()
    local_file.write_bytes(b"local")

    repo = _repo_with_local_dir(tmp_path, revision=None)

    with pytest.raises(huggingface_hub.utils.OfflineModeIsEnabled, match="download_files"):
        repo.download_files(["features/chunk_0.zip"])


def test_revisionless_offline_repo_uses_materialized_local_dir_file(tmp_path, monkeypatch):
    _set_offline_mode(monkeypatch, True)
    local_file = tmp_path / "features.csv"
    local_file.write_bytes(b"local")

    repo = _repo_with_local_dir(tmp_path, revision=None)

    assert repo.revision is None
    assert repo.download_file("features.csv") == str(local_file)
    with pytest.raises(FileNotFoundError, match="missing.csv"):
        repo.download_file("missing.csv")


def test_revisionless_offline_repo_without_local_dir_still_requires_revision(monkeypatch):
    _set_offline_mode(monkeypatch, True)

    with pytest.raises(huggingface_hub.utils.OfflineModeIsEnabled, match="revision"):
        HfRepoInterface(repo_id="org/repo", repo_type="dataset", revision=None)


def test_offline_repo_with_revision_warns_and_falls_back_to_cache(
    tmp_path, monkeypatch, caplog
):
    _set_offline_mode(monkeypatch, True)
    cache_file = tmp_path / "cache_file"
    cache_file.write_bytes(b"cache")

    def try_to_load_from_cache(filename, **kwargs):
        assert kwargs["revision"] == "abcdef"
        return str(cache_file) if filename == "features/missing.zip" else None

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", try_to_load_from_cache)

    with caplog.at_level(logging.WARNING):
        repo = _repo_with_local_dir(tmp_path, revision="abcdef")

    assert "without validating they match that revision" in caplog.text
    assert "fall back to the versioned Hugging Face cache" in caplog.text
    assert repo.download_file("features/missing.zip") == str(cache_file)
