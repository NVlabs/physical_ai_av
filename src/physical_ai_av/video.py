# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
import abc
import collections
import io
import logging

import av
import numpy as np

logger = logging.getLogger(__name__)


class VideoReader(abc.ABC):
    """Abstract base class for video readers."""

    def __init__(
        self,
        video_data: io.BytesIO,
        timestamps: np.ndarray | None = None,
        thread_count: int = 1,
    ):
        """Initialize the reader.

        Args:
            video_data: The video data to read.
            timestamps: Timestamps for each frame in the video.
            thread_count: Number of threads to use for decoding.

        Raises:
            ValueError: If the timestamps are not strictly increasing.
        """
        self.timestamps = timestamps
        if timestamps is not None and np.diff(timestamps).min() <= 0:
            raise ValueError("Timestamps must be strictly increasing")

    def decode_images_from_timestamps(self, requested_timestamps: np.ndarray) -> np.ndarray:
        """Decode images closest before the requested timestamps.

        Args:
            requested_timestamps: Timestamps to decode.

        Returns:
            An array of shape (N, H, W, C) containing the decoded frames.
            An array of shape (N,) containing the timestamps of the decoded frames.

        Raises:
            ValueError: If the requested timestamps are empty.
            ValueError: If the timestamps of the video are not provided.
        """
        if len(requested_timestamps) == 0:
            raise ValueError("Requested timestamps must be non-empty")
        if self.timestamps is None:
            raise ValueError("Timestamps must be provided")
        if self.timestamps.ndim != 1:
            raise ValueError("Timestamps must be 1D")
        frame_idxs = self._get_frame_idxs_from_timestamps(requested_timestamps)
        images = self.decode_images_from_frame_indices(frame_idxs)
        return images, self.timestamps[frame_idxs]

    @abc.abstractmethod
    def decode_images_from_frame_indices(self, frame_indices: np.ndarray) -> np.ndarray:
        """Decode images from frame indices.

        Args:
            frame_indices: Frame indices to decode.

        Returns:
            image: An array of shape (N, H, W, C) containing the decoded frames.
        """
        pass

    def _get_frame_idxs_from_timestamps(self, requested_timestamps: np.ndarray) -> np.ndarray:
        """Get the frame indices of the closest frames before each timestamp.

        Args:
            requested_timestamps: Timestamps to decode.

        Returns:
            frame_idxs: The frame indices of the closest frames before each timestamp.

        Raises:
            ValueError: If the requested timestamps are not within the range of timestamps.
        """
        if not (
            requested_timestamps.min() >= self.timestamps.min()
            and requested_timestamps.max() <= self.timestamps.max()
        ):
            raise ValueError(
                "Requested timestamps must be within the range of timestamps:\n"
                f"{requested_timestamps.min()=}, {requested_timestamps.max()=}\n"
                f"{self.timestamps.min()=}, {self.timestamps.max()=}"
            )
        # compute the frame index of closest frames before each timestamp
        return np.searchsorted(self.timestamps, requested_timestamps, side="right") - 1

    def close(self):
        """Close the video reader."""
        pass


class SeekVideoReader(VideoReader):
    """Reusable random-access reader backed by PyAV/FFmpeg.

    NOTE: This reader works only on constant frame-rate videos.
    """

    def __init__(
        self,
        video_data: io.BytesIO,
        timestamps: np.ndarray | None = None,
        thread_count: int = 1,
    ):
        """Initialize the reader.

        Args:
            video_data: video data to read.
            timestamps: Timestamps for each frame in the video.
            thread_count: Number of threads to use for decoding.
        """
        super().__init__(video_data, timestamps, thread_count)
        video_data.seek(0)
        self.container = av.open(video_data)
        # About thread_counts and thread_type
        # thread_counts:
        #   0 means let ffmpeg decide how many threads to use to decode the video
        #       This is usually means using `av_cpu_count()` and can be not optimal in
        #       multi-processing cases.
        #   >0 means use the specified number of threads
        # thread_type:
        #   "AUTO" means let PyAV decide what kind of multi-threading to use
        #   "SLICE": Decode more than one part of a single frame at once
        #   "FRAME": Decode more than one frame at once
        self.container.streams.video[0].thread_type = av.codec.context.ThreadType.AUTO
        self.container.streams.video[0].thread_count = thread_count
        self.stream = next(s for s in self.container.streams if s.type == "video")
        self.start_time_pts = self.stream.start_time
        self.time_base = self.stream.time_base
        self.fps = self.stream.average_rate
        self.pts_per_frame = int(1 / self.fps / self.time_base)
        # Build a list of **packet PTS values** for every key‑frame once.
        self._key_pts = self._build_keyframe_index()
        logger.debug("key_pts: %s", self._key_pts.tolist())
        logger.debug("start_time_pts: %s", self.start_time_pts)
        logger.debug("time_base: %s", self.time_base)
        logger.debug("fps: %s", self.fps)
        logger.debug("pts_per_frame: %s", self.pts_per_frame)

    def close(self):
        """Close the video reader."""
        if getattr(self, "container", None) is not None:
            self.container.close()
            self.container = None

    def _build_keyframe_index(self):
        """Build a list of **packet PTS values** for every key-frame once.

        Note that this operation is not costly as we are not decoding anything from the
        container.

        We seek to the first key-frame then decode the frames from there.
        """
        key_pts = []
        for pkt in self.container.demux(self.stream):
            if pkt.is_keyframe and pkt.pts is not None:
                key_pts.append(pkt.pts)
        key_pts.sort()
        # Rewind so the first real read starts from the beginning.
        self.container.seek(self.start_time_pts, any_frame=True, backward=True, stream=self.stream)
        return np.array(key_pts)

    def _frame_to_pts(self, frame_idx: int | np.ndarray) -> int | np.ndarray:
        """Convert a frame number to the closest presentation timestamp in stream timebase units."""
        return self.start_time_pts + self.pts_per_frame * frame_idx

    def _prev_key_pts(self, target_pts: np.ndarray) -> np.ndarray:
        """Presentation timestamp of the nearest **preceding** key-frame.

        Args:
            target_pts: Target presentation timestamps.

        Returns:
            prev_key_pts: Presentation timestamps of the nearest **preceding** key-frames.

        Raises:
            ValueError: If any of the target_pts is before the first key-frame.
        """
        idx = np.searchsorted(self._key_pts, target_pts, side="right") - 1
        if not np.all(idx >= 0):
            raise ValueError("All target_pts must be after the first key-frame")
        return self._key_pts[idx]

    def decode_images_from_frame_indices(self, frame_indices: np.ndarray) -> np.ndarray:
        """Decode images from frame indices.

        Args:
            frame_indices: Frame indices to decode.

        Returns:
            An array of shape (N, H, W, C) containing the decoded frames.

        Raises:
            ValueError: If the frame indices are not int64.
            ValueError: If the video reader is closed.
        """
        if getattr(self, "container", None) is None:
            raise ValueError("Video reader is closed")
        if not frame_indices.dtype == np.int64:
            raise ValueError("Frame indices must be int64")
        unique_frame_idxs = np.sort(np.unique(frame_indices))
        target_pts = self._frame_to_pts(unique_frame_idxs)
        collected: dict[int, np.ndarray] = {}

        # Make sure the first frame is decoded to avoid potential corruption.
        # TODO: investigate whether this is still necessary.
        if self._key_pts[0] > self.start_time_pts and 0 not in unique_frame_idxs:
            self.container.seek(
                self.start_time_pts, any_frame=True, backward=True, stream=self.stream
            )
            next(self.container.decode(video=0))

        def _loop_decode(target_frame_idxs: set[int]):
            count = 0
            for frame in self.container.decode(video=0):
                if frame.pts is None:
                    continue
                cur_frame_idx = int(round(frame.pts / self.pts_per_frame))
                if cur_frame_idx in target_frame_idxs:
                    collected[cur_frame_idx] = frame.to_ndarray(format="rgb24")
                    count += 1
                if count == len(target_frame_idxs):
                    break

        # Special handling in the case that the first key-frame is not the first frame (should not
        # be the case).
        frames_before_first_key_frame = unique_frame_idxs[target_pts < self._key_pts[0]]
        if len(frames_before_first_key_frame) > 0:
            # Seek to the very beginning, note that here we need to set any_frame=True
            self.container.seek(
                self.start_time_pts, any_frame=True, backward=True, stream=self.stream
            )
            _loop_decode(frames_before_first_key_frame)

        # compute what key-frames we need to seek to
        frames_after_first_key_frame = unique_frame_idxs[target_pts >= self._key_pts[0]]
        target_pts_after_first_key_frame = target_pts[target_pts >= self._key_pts[0]]
        prev_key_pts = self._prev_key_pts(target_pts_after_first_key_frame)
        key_pts_info = collections.defaultdict(set)
        for target_frame, prev_key_pt in zip(frames_after_first_key_frame, prev_key_pts):
            key_pts_info[int(prev_key_pt)].add(int(target_frame))
        for prev_key_pt, target_frames in key_pts_info.items():
            self.container.seek(prev_key_pt, any_frame=False, backward=True, stream=self.stream)
            _loop_decode(target_frames)
        if len(collected) != len(unique_frame_idxs):
            raise ValueError(
                "Some requested frame numbers are not decoded. "
                f"Requested frame indices: {frame_indices.tolist()}\n"
                f"Decoded frame indices: {list(collected.keys())}"
            )

        # order and stack
        collected = [collected[i] for i in unique_frame_idxs]
        stacked = np.stack(collected)
        # If resulted frames are not sorted / unique, re‑order to match it:
        if not np.all(np.diff(frame_indices) > 0):
            order = {frame_idx: i for i, frame_idx in enumerate(unique_frame_idxs)}
            stacked = stacked[[order[frame_idx] for frame_idx in frame_indices]]
        return stacked
