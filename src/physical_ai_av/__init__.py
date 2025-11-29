# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
import importlib.metadata
import logging

from .utils.hf_interface import HfRepoInterface
from .dataset import PhysicalAIAVDatasetInterface

logging.getLogger(__name__).addHandler(logging.NullHandler())

__version__ = importlib.metadata.version("physical_ai_av")
