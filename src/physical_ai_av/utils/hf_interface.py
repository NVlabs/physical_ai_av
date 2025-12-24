# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
import contextlib
import io
import logging
import pathlib
from typing import Any, Iterator

import huggingface_hub
import huggingface_hub.hf_api
import huggingface_hub.utils as hf_utils
import tqdm.contrib.concurrent
from huggingface_hub.utils import tqdm as hf_tqdm

try:
    # huggingface-hub>=1.0
    from huggingface_hub import DryRunFileInfo
except ImportError:
    # huggingface-hub<1.0
    DryRunFileInfo = Any

logger = logging.getLogger(__name__)


class HfRepoInterface:
    """Interface for interacting with individual Hugging Face repositories.

    This class is a thin wrapper around Hugging Face Hub's
        - [`HfApi`](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_api) and
        - [`HfFileSystem`](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_file_system)
    with utility methods added for working with very large (in terms of both file count and total
    file size) repositories.

    Attributes:
        api (`huggingface_hub.HfApi`): Hugging Face Hub API client.
        fs (`huggingface_hub.HfFileSystem`): Hugging Face Hub file system interface.
        repo_id (`str`): A user or an organization name and a repo name separated by a `/`.
        repo_type (`str | None`): `"dataset"` if the repo is a dataset, `"space"` if the repo is a
            space, `None` or `"model"` if the repo is a model.
        revision (`str`): A Git revision id, which can be a branch name, a tag, or a commit hash
            (if not supplied at initialization, the latest commit hash on `main` will be used).
        repo_snapshot_info (`dict`): A dictionary containing the `repo_id`, `repo_type`, and
            `revision` for simpler unpacking into `api` method calls.
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
    """

    def __init__(
        self,
        repo_id: str,
        repo_type: str | None = None,
        revision: str | None = None,
        *,
        token: str | bool | None = None,
        cache_dir: str | pathlib.Path | None = None,
        local_dir: str | pathlib.Path | None = None,
        confirm_download_threshold_gb: float = 10.0,
    ) -> None:
        self.token = token
        self.api = huggingface_hub.HfApi(token=self.token)
        self.fs = huggingface_hub.HfFileSystem(token=self.token)

        self.repo_id = repo_id
        self.repo_type = repo_type
        if revision is None:
            for branch in self.api.list_repo_refs(
                repo_id=self.repo_id,
                repo_type=self.repo_type,
            ).branches:
                if branch.name == "main":
                    revision = branch.target_commit
                    break
            else:
                raise ValueError("No `revision` specified and no `main` branch found on remote.")
        self.revision = revision
        self.repo_snapshot_info = {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "revision": self.revision,
        }

        self.cache_dir = cache_dir
        self.local_dir = local_dir
        self.confirm_download_threshold_gb = confirm_download_threshold_gb

    def __repr__(self) -> str:
        """Returns a string representation of the repo snapshot identifying information."""
        return f"""{self.__class__.__name__}({
            ", ".join(
                f"{attr}={attr_value!r}" for attr, attr_value in self.repo_snapshot_info.items()
            )
        })"""

    def _confirm_download(self, size: int) -> bool:
        """Returns `True` iff the download `size` (bytes) is small enough or the user confirms."""
        return size <= self.confirm_download_threshold_gb * 1024**3 or (
            input(f"Download an additional (uncached) {size / 1024**3:.2f} GB? [Y/n]: ")
            .strip()
            .lower()
            in ("y", "")
        )

    def is_file_cached(self, filename: str) -> bool:
        """Returns `True` iff the file specified by `filename` is cached."""
        return isinstance(
            huggingface_hub.try_to_load_from_cache(
                filename=filename, cache_dir=self.cache_dir, **self.repo_snapshot_info
            ),
            str,
        )

    def download_files(
        self, files: list[str | huggingface_hub.hf_api.RepoFile], max_workers: int = 8, **kwargs
    ) -> list[str | DryRunFileInfo]:
        """Downloads `files`; see `HfApi.hf_hub_download` for more kwargs."""
        # NOTE: `HfApi.snapshot_download` (e.g., with a filter) seems to struggle with large repos,
        # so here we implement our own multi-file downloader. We should check periodically to see
        # if the upstream `huggingface_hub` utilies have improved for this use case.
        if kwargs.get("subfolder", None) is not None:
            raise ValueError(
                "`subfolder` must be `None`, i.e., `files` must be relative to the repo root."
            )
        tqdm_class = kwargs.get("tqdm_class", None) or hf_tqdm

        if any(isinstance(file, str) for file in files):
            try:
                GET_PATHS_INFO_BATCH_SIZE = huggingface_hub._commit_api.FETCH_LFS_BATCH_SIZE
            except AttributeError:
                GET_PATHS_INFO_BATCH_SIZE = 500
                logger.warning(
                    "`huggingface_hub._commit_api.FETCH_LFS_BATCH_SIZE` not found; probably this "
                    "means something upstream has changed and our brittle usage of undocumented "
                    "HF Hub internals may no longer be necessary. Please file an issue!"
                )
            file_info = {}
            str_files = [file for file in files if isinstance(file, str)]
            for offset in range(0, len(str_files), GET_PATHS_INFO_BATCH_SIZE):
                file_info.update(
                    {
                        path_info.path: path_info
                        for path_info in self.api.get_paths_info(
                            paths=str_files[offset : offset + GET_PATHS_INFO_BATCH_SIZE],
                            **self.repo_snapshot_info,
                        )
                    }
                )
            files = [file_info[file] if isinstance(file, str) else file for file in files]
        if not all(isinstance(file, huggingface_hub.hf_api.RepoFile) for file in files):
            not_a_file = next(
                file for file in files if not isinstance(file, huggingface_hub.hf_api.RepoFile)
            )
            raise ValueError(f"{not_a_file!r} is not a file.")

        is_cached_bools = [self.is_file_cached(file.path) for file in files]
        total_size = sum(file.size for file in files)
        cached_size = sum(file.size for file, is_cached in zip(files, is_cached_bools) if is_cached)
        download_size = total_size - cached_size
        logger.info(
            f"Total download size: {total_size / 1024**3:.2f} GB, {len(files)} files "
            f"(cached: {cached_size / 1024**3:.2f} GB, {sum(is_cached_bools)} files)"
        )

        if self._confirm_download(download_size):
            if huggingface_hub.__version__ >= "1.1.3":
                bytes_progress = tqdm_class(
                    desc="Downloading",
                    disable=hf_utils.is_tqdm_disabled(log_level=logger.getEffectiveLevel()),
                    total=total_size,
                    initial=cached_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    **(
                        {"name": f"{self.__class__.__name__}.download_files"}
                        if tqdm_class is hf_tqdm
                        else {}
                    ),
                )

                class _AggregatedTqdm:
                    def __init__(self, *args, **kwargs):
                        pass

                    def __enter__(self):
                        return self

                    def __exit__(self, exc_type, exc_value, traceback):
                        pass

                    def update(self, n: int | float | None = 1) -> None:
                        bytes_progress.update(n)

            def _download_file(
                file: huggingface_hub.hf_api.RepoFile,
            ) -> str | DryRunFileInfo:
                return self.api.hf_hub_download(
                    filename=file.path,
                    cache_dir=self.cache_dir,
                    local_dir=self.local_dir,
                    **(
                        {"tqdm_class": _AggregatedTqdm}
                        if huggingface_hub.__version__ >= "1.1.3"
                        else {}
                    ),
                    **self.repo_snapshot_info,
                    **kwargs,
                )

            tqdm_desc = f"Fetching {len(files)} files ({sum(is_cached_bools)} cached)"
            if kwargs.get("dry_run", False):
                tqdm_desc = "[dry-run] " + tqdm_desc
            return tqdm.contrib.concurrent.thread_map(
                _download_file,
                files,
                desc=tqdm_desc,
                max_workers=max_workers,
                tqdm_class=tqdm_class,
            )

    def download_file(self, filename: str, **kwargs) -> str | DryRunFileInfo:
        """Downloads `filename`; see `HfApi.hf_hub_download` for more kwargs."""
        if self.is_file_cached(filename) or self._confirm_download(
            self.api.get_paths_info(paths=[filename], **self.repo_snapshot_info)[0].size
        ):
            return self.api.hf_hub_download(
                filename=filename,
                cache_dir=self.cache_dir,
                local_dir=self.local_dir,
                **self.repo_snapshot_info,
                **kwargs,
            )

    def download_repo_tree(
        self, path_in_repo: str, recursive: bool = True, **kwargs
    ) -> list[str | DryRunFileInfo]:
        """Downloads `path_in_repo`; see `HfApi.hf_hub_download` for more kwargs."""
        return self.download_files(
            [
                repo_item
                for repo_item in self.api.list_repo_tree(
                    path_in_repo=path_in_repo,
                    recursive=recursive,
                    **self.repo_snapshot_info,
                )
                if isinstance(repo_item, huggingface_hub.hf_api.RepoFile)
            ],
            **kwargs,
        )

    @contextlib.contextmanager
    def open_file(
        self, filename: str, mode="rb", maybe_stream: bool = False
    ) -> Iterator[io.BytesIO]:
        """Opens `filename` from the cache if it exists, or streams it from HF Hub."""
        filepath = huggingface_hub.try_to_load_from_cache(
            filename=filename,
            cache_dir=self.cache_dir,
            **self.repo_snapshot_info,
        )
        if isinstance(filepath, str):
            with open(filepath, mode) as f:
                yield f
        elif maybe_stream:
            with self.fs.open(
                ("datasets/" if self.repo_type == "dataset" else "") + f"{self.repo_id}/{filename}",
                mode,
            ) as f:
                yield f
        else:
            raise FileNotFoundError(
                f"{filename=} not found in cache; "
                "set `maybe_stream=True` to enable streaming from HF Hub."
            )

class OfflineHfRepoInterface:
    """Fully offline simulation of Hugging Face repository interface.

    This class is a thin wrapper around Hugging Face Hub's
        - [`HfApi`](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_api) and
        - [`HfFileSystem`](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_file_system)
    with utility methods added for working with very large (in terms of both file count and total
    file size) repositories.

    Attributes:
        api (`huggingface_hub.HfApi`): Hugging Face Hub API client.
        fs (`huggingface_hub.HfFileSystem`): Hugging Face Hub file system interface.
        repo_id (`str`): A user or an organization name and a repo name separated by a `/`.
        repo_type (`str | None`): `"dataset"` if the repo is a dataset, `"space"` if the repo is a
            space, `None` or `"model"` if the repo is a model.
        revision (`str`): A Git revision id, which can be a branch name, a tag, or a commit hash
            (if not supplied at initialization, the latest commit hash on `main` will be used).
        repo_snapshot_info (`dict`): A dictionary containing the `repo_id`, `repo_type`, and
            `revision` for simpler unpacking into `api` method calls.
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
    """
    def __init__(
        self,
        repo_id: str,
        repo_type: str | None = None,
        revision: str | None = None,
        *,
        token: str | bool | None = None,
        cache_dir: str | pathlib.Path | None = None,
        local_dir: str | pathlib.Path | None = None,
        confirm_download_threshold_gb: float = 10.0,
    ) -> None:
        self.token = token
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.revision = revision or "local"
        
        if local_dir is None:
            raise ValueError("local_dir must be provided for offline mode")
        
        self.local_dir = pathlib.Path(local_dir)
        if not self.local_dir.exists():
            raise FileNotFoundError(f"Local directory not found: {self.local_dir}")
            
        self.cache_dir = cache_dir
        self.confirm_download_threshold_gb = confirm_download_threshold_gb
        
        self.repo_snapshot_info = {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "revision": self.revision,
        }
        
        logger.info(f"Offline mode enabled. Using local directory: {self.local_dir}")

    def is_file_cached(self, filename: str) -> bool:
        """check local files"""
        local_path = self.local_dir / filename
        return local_path.exists()

    def download_file(self, filename: str, **kwargs) -> str:
        """Simulate downloading files, but actually return the local path"""
        local_path = self.local_dir / filename
        
        if local_path.exists():
            return str(local_path)
            
        possible_paths = [
            local_path,
            self.local_dir / "data" / filename,
            self.local_dir / filename.split("/")[-1],
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        raise FileNotFoundError(
            f"File '{filename}' not found in local directory {self.local_dir}. "
            f"Please make sure all data files are available locally."
        )

    def download_files(self, files: list[str], max_workers: int = 8, **kwargs) -> list[str]:
        results = []
        for filename in files:
            try:
                result = self.download_file(filename, **kwargs)
                results.append(result)
                logger.debug(f"Successfully loaded: {filename}")
            except FileNotFoundError as e:
                logger.warning(f"Failed to load: {filename} - {e}")
                if not kwargs.get("dry_run", False):
                    raise
        return results

    def download_repo_tree(
        self, path_in_repo: str, recursive: bool = True, **kwargs
    ) -> list[str]:
        dir_path = self.local_dir / path_in_repo
        if not dir_path.exists():
            return []
        
        files = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in dir_path.glob(pattern):
            if file_path.is_file():
                rel_path = file_path.relative_to(self.local_dir)
                files.append(str(file_path))
        
        return files

    @contextlib.contextmanager
    def open_file(
        self, filename: str, mode: str = "rb", maybe_stream: bool = False
    ) -> Iterator[io.BytesIO]:
        """open local file"""
        filepath = self.download_file(filename)
        
        with open(filepath, mode) as f:
            yield f

    def __repr__(self) -> str:
        return f"OfflineHfRepoInterface(local_dir={self.local_dir!r})"
