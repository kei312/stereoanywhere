"""Helpers for memory aware MapReduce execution."""
from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class TilingParameters:
    tile_size: int
    overlap: int


def available_vram_mb(device: int = 0) -> float:
    """Return the approximate free VRAM on the given CUDA device."""

    if not torch.cuda.is_available():
        return 0.0

    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    stats = torch.cuda.memory_stats(device)
    reserved = stats.get("reserved_bytes.all", 0)
    allocated = stats.get("allocated_bytes.all", 0)
    total = torch.cuda.get_device_properties(device).total_memory
    free = max(total - max(reserved, allocated), 0)
    return free / (1024**2)


def select_tiling_parameters(
    default_tile: int = 1024,
    default_overlap: int = 256,
    min_tile: int = 256,
    device: int = 0,
) -> TilingParameters:
    """Choose tile size heuristically based on available VRAM."""

    free_mb = available_vram_mb(device)

    if free_mb <= 0:
        return TilingParameters(tile_size=min_tile, overlap=min_tile // 4)

    if free_mb < 2048:  # < 2 GB
        tile = max(min_tile, 512)
    elif free_mb < 4096:
        tile = max(min_tile, 768)
    elif free_mb < 6144:
        tile = max(min_tile, 896)
    else:
        tile = default_tile

    overlap = min(tile // 4, default_overlap)
    return TilingParameters(tile_size=tile, overlap=overlap)


def log_memory_snapshot(prefix: str = "") -> None:
    """Print current GPU memory usage for debugging."""

    if not torch.cuda.is_available():
        print(f"{prefix}CUDA not available – running on CPU")
        return

    device = torch.cuda.current_device()
    stats = torch.cuda.memory_stats(device)
    allocated = stats.get("allocated_bytes.all", 0) / (1024**2)
    reserved = stats.get("reserved_bytes.all", 0) / (1024**2)
    free = available_vram_mb(device)
    print(
        f"{prefix} GPU memory – allocated: {allocated:.1f} MB, "
        f"reserved: {reserved:.1f} MB, approx free: {free:.1f} MB"
    )

    gc.collect()
    torch.cuda.empty_cache()
