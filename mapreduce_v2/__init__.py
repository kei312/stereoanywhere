"""MapReduce utilities for StereoAnywhere.

This module provides wrappers that allow tiled inference, optional
CPU offloading, and heuristics for handling non-Lambertian regions
without modifying the original StereoAnywhere implementation.
"""

from .tile_wrapper import TileWrapper
from .tiled_inference import MapReduceInference
from .non_lambertian import NonLambertianProcessor
from .cpu_offload_wrapper import CPUOffloadWrapper
from .memory_utils import select_tiling_parameters, log_memory_snapshot
from .tile_presets import (
    TilePreset,
    TILE_PRESETS,
    get_preset,
    list_presets,
    get_preset_for_dataset,
    create_custom_preset,
)

__all__ = [
    "TileWrapper",
    "MapReduceInference",
    "NonLambertianProcessor",
    "CPUOffloadWrapper",
    "select_tiling_parameters",
    "log_memory_snapshot",
    "TilePreset",
    "TILE_PRESETS",
    "get_preset",
    "list_presets",
    "get_preset_for_dataset",
    "create_custom_preset",
]
