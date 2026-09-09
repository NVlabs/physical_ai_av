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
        revision (`str | None`): A Git revision id, which can be a branch name, a tag, or a
            commit hash. If not supplied at initialization while online, the latest commit hash on
            `main` will be used. If not supplied while offline, `local_dir` must be set and only
            materialized files under `local_dir` can be used.
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
        requested_revision = revision
        self.token = token
        self.api = huggingface_hub.HfApi(token=self.token)
        self.fs = huggingface_hub.HfFileSystem(token=self.token)
        if huggingface_hub.__version__ >= "1.2.1":
            self.is_offline_mode = huggingface_hub.is_offline_mode()
        else:
            self.is_offline_mode = huggingface_hub.constants.HF_HUB_OFFLINE

        self.repo_id = repo_id
        self.repo_type = repo_type
        self.cache_dir = cache_dir
        self.local_dir = local_dir
        self.confirm_download_threshold_gb = confirm_download_threshold_gb
        if revision is None:
            if self.is_offline_mode and self.local_dir is None:
                raise hf_utils.OfflineModeIsEnabled(
                    "`revision` must be specified when offline mode is enabled."
                )
            elif not self.is_offline_mode:
                for branch in self.api.list_repo_refs(
                    repo_id=self.repo_id,
                    repo_type=self.repo_type,
                ).branches:
                    if branch.name == "main":
                        revision = branch.target_commit
                        break
                else:
                    raise ValueError(
                        "No `revision` specified and no `main` branch found on remote."
                    )
        elif self.local_dir is not None:
            logger.warning(
                "`local_dir` is set with `revision=%r`; files already present in `local_dir` "
                "will be used without validating they match that revision. Missing files will %s.",
                requested_revision,
                "fall back to the versioned Hugging Face cache"
                if self.is_offline_mode
                else "be downloaded from the requested revision",
            )
        self.revision = revision
        self.repo_snapshot_info = {
            "repo_id": self.repo_id,
            "repo_type": self.repo_type,
            "revision": self.revision,
        }

    def _get_repo_filepath(
        self, filename: str, subfolder: str | pathlib.Path | None = None
    ) -> pathlib.PurePosixPath:
        """Returns the repo-relative filepath for a filename and optional subfolder."""
        relative_filepath = pathlib.PurePosixPath(filename)
        if relative_filepath.is_absolute() or ".." in relative_filepath.parts:
            raise ValueError(f"{filename=!r} must be a repo-relative path.")

        if subfolder is None:
            return relative_filepath

        relative_subfolder = pathlib.PurePosixPath(subfolder)
        if relative_subfolder.is_absolute() or ".." in relative_subfolder.parts:
            raise ValueError(f"{subfolder=!r} must be a repo-relative path.")

        return relative_subfolder / relative_filepath

    def _get_local_filepath(
        self, filename: str, subfolder: str | pathlib.Path | None = None
    ) -> pathlib.Path | None:
        """Returns `filename` resolved under `local_dir`, if `local_dir` is configured."""
        if self.local_dir is None:
            return None

        relative_filepath = self._get_repo_filepath(filename, subfolder=subfolder)
        return pathlib.Path(self.local_dir).joinpath(*relative_filepath.parts)

    def _get_materialized_filepath(
        self, filename: str, subfolder: str | pathlib.Path | None = None
    ) -> pathlib.Path | None:
        """Returns the local materialized filepath for `filename`, if it exists."""
        local_filepath = self._get_local_filepath(filename, subfolder=subfolder)
        if local_filepath is not None and local_filepath.is_file():
            return local_filepath
        return None

    def _get_cached_filepath(
        self, filename: str, subfolder: str | pathlib.Path | None = None
    ) -> pathlib.Path | None:
        """Returns the cached filepath for `filename`, if it exists for this revision."""
        if self.revision is None:
            return None

        relative_filepath = self._get_repo_filepath(filename, subfolder=subfolder)
        filepath = huggingface_hub.try_to_load_from_cache(
            filename=relative_filepath.as_posix(),
            cache_dir=self.cache_dir,
            **self.repo_snapshot_info,
        )
        if isinstance(filepath, str):
            return pathlib.Path(filepath)
        return None

    def _resolve_local_filepath(
        self,
        filename: str,
        subfolder: str | pathlib.Path | None = None,
        *,
        allow_cache: bool = True,
    ) -> pathlib.Path | None:
        """Returns the best available local filepath, preferring `local_dir` over the cache."""
        materialized_filepath = self._get_materialized_filepath(filename, subfolder=subfolder)
        if materialized_filepath is not None:
            return materialized_filepath
        if allow_cache:
            return self._get_cached_filepath(filename, subfolder=subfolder)
        return None

    def _raise_file_not_available_offline(
        self, filename: str, subfolder: str | pathlib.Path | None = None
    ) -> None:
        """Raises the appropriate error when `filename` cannot be downloaded."""
        relative_filepath = self._get_repo_filepath(filename, subfolder=subfolder)
        if self.revision is None and self.local_dir is not None:
            raise FileNotFoundError(
                f"{relative_filepath.as_posix()!r} not found in local_dir and "
                "offline mode is enabled."
            )
        raise hf_utils.OfflineModeIsEnabled(
            f"{relative_filepath.as_posix()!r} not found locally and offline mode is enabled."
        )

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
        return self._get_cached_filepath(filename) is not None

    def is_file_materialized(self, filename: str) -> bool:
        """Returns `True` iff the file specified by `filename` exists in local_dir."""
        return self._get_materialized_filepath(filename) is not None

    def is_file_available_locally(self, filename: str) -> bool:
        """Returns `True` iff the file specified by `filename` exists locally."""
        return self._resolve_local_filepath(filename) is not None

    def download_files(
        self, files: list[str | huggingface_hub.hf_api.RepoFile], max_workers: int = 8, **kwargs
    ) -> list[str | DryRunFileInfo]:
        """Downloads `files` from the Hugging Face Hub; see `HfApi.hf_hub_download`.

        This is an online prefetch/download API, not the local serving path for a
        materialized `local_dir` dataset. It requires Hub access to discover and download files.
        Use `open_file()` or higher-level feature accessors to read files already present in
        `local_dir`.
        """
        if kwargs.get("subfolder", None) is not None:
            raise ValueError(
                "`subfolder` must be `None`, i.e., `files` must be relative to the repo root."
            )

        if self.is_offline_mode:
            raise hf_utils.OfflineModeIsEnabled(
                "`download_files()` requires online mode because it fetches file metadata from "
                "the Hugging Face Hub. Use `open_file()` or higher-level feature accessors to "
                "read files already present in `local_dir`."
            )

        # NOTE: `HfApi.snapshot_download` (e.g., with a filter) seems to struggle with large repos,
        # so here we implement our own multi-file downloader. We should check periodically to see
        # if the upstream `huggingface_hub` utilities have improved for this use case.
        tqdm_class = kwargs.pop("tqdm_class", None) or hf_tqdm

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

                    def close(self) -> None:
                        pass

                    def update(self, n: int | float | None = 1) -> None:
                        bytes_progress.update(n)

                    def set_postfix_str(self, postfix: str, refresh: bool = False) -> None:
                        pass

                    def update_transfer(self, n: int | float | None = 1) -> None:
                        # Newer huggingface_hub Xet downloads use separate transfer and
                        # reconstruction progress signals. This wrapper only owns one aggregate
                        # byte bar, so `update()` is the only method that advances it; advancing
                        # here as well would double-count HTTP downloads that emit both signals.
                        pass

                    def set_transfer_postfix_str(
                        self, postfix: str, refresh: bool = False
                    ) -> None:
                        pass

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
            downloaded_filepaths = tqdm.contrib.concurrent.thread_map(
                _download_file,
                files,
                desc=tqdm_desc,
                max_workers=max_workers,
                tqdm_class=tqdm_class,
            )
            return downloaded_filepaths

    def download_file(self, filename: str, **kwargs) -> str | DryRunFileInfo | None:
        """Downloads `filename`; see `HfApi.hf_hub_download` for more kwargs.

        Similar to `HfApi.hf_hub_download`, if the file is cached it will not be re-downloaded
        (unless the `force_download=True` kwarg is passed) and the existing cached filepath will be
        returned. In particular, this function is safe to use with the environment variable
        `HF_HUB_OFFLINE=1` to get locations of cached files (provided they've been pre-downloaded)
        while offline.

        Args:
            filename (`str`): The filename (relative to the repo root) to download.
            **kwargs: Additional keyword arguments to pass to `HfApi.hf_hub_download`.

        Returns:
            `str | DryRunFileInfo | None`: The path to the downloaded/cached file, dry-run info (if
                the `dry_run=True` kwarg is passed), or `None` (if the file download is skipped).
        """
        subfolder = kwargs.get("subfolder", None)
        force_download = kwargs.get("force_download", False)
        if not force_download:
            local_filepath = self._resolve_local_filepath(
                filename,
                subfolder=subfolder,
                allow_cache=self.is_offline_mode,
            )
            if local_filepath is not None:
                return str(local_filepath)

        if self.is_offline_mode:
            self._raise_file_not_available_offline(filename, subfolder=subfolder)

        repo_filepath = self._get_repo_filepath(filename, subfolder=subfolder)
        if self._get_cached_filepath(filename, subfolder=subfolder) is None:
            path_info = self.api.get_paths_info(
                paths=[repo_filepath.as_posix()], **self.repo_snapshot_info
            )
            if len(path_info) == 0:
                raise ValueError(
                    f"{repo_filepath.as_posix()!r} not found in {self.repo_snapshot_info}."
                )
            path_info = path_info[0]
            if not self._confirm_download(path_info.size):
                logger.info(f"Skipping download of {repo_filepath.as_posix()!r}.")
                return
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
        """Downloads `path_in_repo` from the Hugging Face Hub.

        This is an online prefetch/download API because it lists the remote repo tree before
        downloading files. It is not the local serving path for a materialized `local_dir` dataset.
        """
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
        """Opens `filename` from `local_dir`/cache if it exists, or streams it from HF Hub."""
        local_filepath = self._resolve_local_filepath(filename)
        if local_filepath is not None:
            with open(local_filepath, mode) as f:
                yield f
            return

        if self.is_offline_mode:
            self._raise_file_not_available_offline(filename)

        if maybe_stream:
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
