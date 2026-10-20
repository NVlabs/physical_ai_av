# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Open local auxiliary Zarr stores and select frames."""

import operator
import tarfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt

try:
    import zarr
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Depth and occupancy readers require Zarr 2; install physical_ai_av[f4d]"
    ) from error


class _TarStore(zarr.storage.Store):
    """Read Zarr members from an uncompressed tar or indexed-tar archive."""

    _writeable = False
    _erasable = False

    def __init__(self, path: Path) -> None:
        self._lock = threading.Lock()
        self._members: dict[str, tuple[int, int]] = {}
        self._file = path.open("rb")
        try:
            with tarfile.open(fileobj=self._file, mode="r:") as archive:
                for member in archive:
                    if member.isfile():
                        self._members[member.name.removeprefix("./")] = (
                            member.offset_data,
                            member.size,
                        )
                    # Keep only offsets, not a second index of every TarInfo object.
                    archive.members.clear()
        except BaseException:
            self._file.close()
            raise

    def __getitem__(self, key: str) -> bytes:
        offset, size = self._members[key]
        with self._lock:
            self._file.seek(offset)
            data = self._file.read(size)
        if len(data) != size:
            raise OSError(f"Truncated tar member: {key}")
        return data

    def __contains__(self, key: object) -> bool:
        return key in self._members

    def __iter__(self) -> Iterator[str]:
        return iter(self._members)

    def __len__(self) -> int:
        return len(self._members)

    def __setitem__(self, key: str, value: bytes) -> None:
        raise PermissionError("Auxiliary tar stores are read-only")

    def __delitem__(self, key: str) -> None:
        raise PermissionError("Auxiliary tar stores are read-only")

    def close(self) -> None:
        """Close the archive file."""
        with self._lock:
            self._file.close()


class _AuxiliaryReader:
    def __init__(self, path: str | Path, signal: str) -> None:
        path = Path(path)
        self._store = zarr.storage.DirectoryStore(path) if path.is_dir() else _TarStore(path)
        self._closed = False
        try:
            root = zarr.open_group(store=self._store, mode="r")
            root_name = root.attrs.get("aux_root_group_name", "annotations")
            self._group = root[root_name][signal]
        except BaseException:
            self.close()
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise ValueError("Reader is closed")

    def close(self) -> None:
        """Close the auxiliary store; repeated calls are safe."""
        self._store.close()
        self._closed = True

    def __enter__(self) -> Self:
        """Return the open reader."""
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Close the reader on context exit."""
        self.close()


def _frame_timestamp(
    timestamps: npt.NDArray[np.int64],
    *,
    frame_index: int | None,
    timestamp_us: int | None,
) -> int:
    if frame_index is not None and timestamp_us is not None:
        raise ValueError("Provide only one of frame_index or timestamp_us")
    if frame_index is None and timestamp_us is None:
        frame_index = 0
    if timestamps.size == 0:
        raise ValueError("No frames are available")
    if frame_index is not None:
        frame_index = operator.index(frame_index)
        if not 0 <= frame_index < len(timestamps):
            raise IndexError(f"Frame index out of range: {frame_index}")
        return int(timestamps[frame_index])
    timestamp_us = operator.index(timestamp_us)
    if timestamp_us < timestamps[0] or timestamp_us > timestamps[-1]:
        raise ValueError("Requested timestamp must be within the range of timestamps")
    return int(timestamps[np.searchsorted(timestamps, timestamp_us, side="right") - 1])
