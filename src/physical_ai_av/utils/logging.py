# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
import logging
import sys
from typing import TextIO

DEFAULT_FMT = "%(asctime)s [%(levelname)-8s] %(name)s (%(funcName)s:%(lineno)d): %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger(__name__)


def setup(
    level: int = logging.INFO,
    stream: TextIO = sys.stdout,
    fmt: str = DEFAULT_FMT,
    datefmt: str = DEFAULT_DATEFMT,
    force_new_handler: bool = False,
) -> None:
    """Sets up basic logging for this package."""
    pkg_logger = logging.getLogger(__name__.split(".")[0])
    if not force_new_handler and any(
        not isinstance(handler, logging.NullHandler) for handler in pkg_logger.handlers
    ):
        logger.info(
            "Logging appears to already be configured; skipping setup "
            "(set `force_new_handler=True` to override)"
        )
        return

    pkg_logger.setLevel(level)
    handler = logging.StreamHandler(stream)
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handler.setFormatter(formatter)
    pkg_logger.addHandler(handler)
    pkg_logger.propagate = False

    logger.info(f"Logging configured by {__name__}.setup()")
