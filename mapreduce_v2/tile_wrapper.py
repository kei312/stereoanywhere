"""Tiled inference wrapper for StereoAnywhere models.

The wrapper accepts any callable that follows the StereoAnywhere
forward signature and executes it on overlapping tiles so that
high-resolution images can be processed on memory constrained GPUs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

Tensor = torch.Tensor


@dataclass
class TileSpec:
    """Metadata describing a single tile."""

    y_start: int
    y_end: int
    x_start: int
    x_end: int

    @property
    def height(self) -> int:
        return self.y_end - self.y_start

    @property
    def width(self) -> int:
        return self.x_end - self.x_start


def _make_blend_weight(height: int, width: int, device: torch.device) -> Tensor:
    """Create a 2D cosine blending mask for smooth tile stitching."""

    if height <= 0 or width <= 0:
        raise ValueError("Tile dimensions must be positive")

    y = torch.linspace(0, 1, height, device=device)
    x = torch.linspace(0, 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    weight_y = torch.sin(torch.pi * torch.clamp(grid_y, 0, 1))
    weight_x = torch.sin(torch.pi * torch.clamp(grid_x, 0, 1))
    weight = torch.clamp(weight_y * weight_x, min=1e-4)
    return weight


class TileWrapper(nn.Module):
    """Wrap a StereoAnywhere-like module to operate on overlapping tiles."""

    def __init__(
        self,
        model: nn.Module,
        tile_size: Optional[int] = None,
        tile_width: Optional[int] = None,
        tile_height: Optional[int] = None,
        overlap: int = 256,
        batch_tiles: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        
        # Xác định sử dụng tile hình chữ nhật hay hình vuông
        use_rectangular = tile_width is not None and tile_height is not None and tile_width > 0 and tile_height > 0
        
        if use_rectangular:
            # Sử dụng tile hình chữ nhật
            if tile_width <= 0 or tile_height <= 0:
                raise ValueError("tile_width and tile_height must be > 0")
            self.tile_width = tile_width
            self.tile_height = tile_height
            self.use_rectangular = True
        else:
            # Sử dụng tile hình vuông (logic cũ)
            if tile_size is None or tile_size <= 0:
                raise ValueError("tile_size must be > 0 when not using rectangular tiles")
            self.tile_width = tile_size
            self.tile_height = tile_size
            self.use_rectangular = False
        
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= min(self.tile_width, self.tile_height):
            raise ValueError("overlap must be smaller than the minimum of tile_width and tile_height")

        self.model = model
        self.overlap = overlap
        self.batch_tiles = batch_tiles
        self._device_override = device

    @property
    def device(self) -> torch.device:
        if self._device_override is not None:
            return self._device_override
        return next(self.model.parameters()).device

    def _enumerate_tiles(self, height: int, width: int) -> List[TileSpec]:
        stride_y = self.tile_height - self.overlap
        stride_x = self.tile_width - self.overlap
        tiles: List[TileSpec] = []

        y = 0
        while y < height:
            y_end = min(y + self.tile_height, height)
            y_start = max(0, y_end - self.tile_height)

            x = 0
            while x < width:
                x_end = min(x + self.tile_width, width)
                x_start = max(0, x_end - self.tile_width)
                tiles.append(TileSpec(y_start=y_start, y_end=y_end, x_start=x_start, x_end=x_end))
                x += stride_x

            y += stride_y

        return tiles

    def forward(
        self,
        left: Tensor,
        right: Tensor,
        mono_left: Optional[Tensor] = None,
        mono_right: Optional[Tensor] = None,
        *args,
        mixed_precision: bool = False,
        global_guidance: Optional[Tensor] = None,
        guidance_weight: float = 0.3,
        **kwargs,
    ) -> Tensor:
        """Run tiled inference and stitch the outputs back together."""

        mixed_precision = kwargs.pop("mixed_precision", False)

        if left.shape != right.shape:
            raise ValueError("Left/right inputs must have identical shape")
        if mono_left is not None:
            if mono_left.shape[0] != left.shape[0] or mono_left.shape[-2:] != left.shape[-2:]:
                raise ValueError("mono_left must share batch and spatial shape with left input")
        if mono_right is not None:
            if mono_right.shape[0] != right.shape[0] or mono_right.shape[-2:] != right.shape[-2:]:
                raise ValueError("mono_right must share batch and spatial shape with right input")

        batch, channels, height, width = left.shape
        if batch != 1:
            raise ValueError("TileWrapper currently supports batch size == 1")

        if height <= self.tile_height and width <= self.tile_width:
            output = self.model(left, right, mono_left, mono_right, *args, **kwargs)
            return self._canonicalize_output(output)

        device = self.device
        tiles = self._enumerate_tiles(height, width)

        stitched = torch.zeros((batch, 1, height, width), device=device, dtype=torch.float32)
        weight_map = torch.zeros_like(stitched)

        # Lưu guidance để sử dụng trong tile processing
        self._global_guidance = global_guidance
        self._guidance_weight = guidance_weight

        if self.batch_tiles:
            outputs = self._process_tiles_batch(
                tiles, left, right, mono_left, mono_right, args, kwargs, mixed_precision
            )
            for tile, disp in zip(tiles, outputs):
                self._accumulate_tile(tile, disp, stitched, weight_map)
        else:
            for tile in tiles:
                disp = self._process_single_tile(
                    tile,
                    left,
                    right,
                    mono_left,
                    mono_right,
                    args,
                    kwargs,
                    mixed_precision,
                )
                self._accumulate_tile(tile, disp, stitched, weight_map)

        stitched = torch.where(weight_map > 0, stitched / torch.clamp(weight_map, min=1e-4), stitched)
        return stitched

    def _canonicalize_output(self, output) -> Tensor:
        """Normalize model outputs to a BCHW disparity tensor."""

        if isinstance(output, (tuple, list)):
            output = output[0]

        if not isinstance(output, torch.Tensor):
            raise TypeError("Model output must be a tensor or tuple/list of tensors")

        if output.dim() == 3:
            output = output.unsqueeze(1)

        if output.dim() != 4:
            raise ValueError("Model output must be BCHW")

        if output.shape[1] != 1:
            raise ValueError("Disparity tensor must have a single channel")

        return -output

    def _process_single_tile(
        self,
        tile: TileSpec,
        left: Tensor,
        right: Tensor,
        mono_left: Optional[Tensor],
        mono_right: Optional[Tensor],
        args: Sequence,
        kwargs: dict,
        mixed_precision: bool,
    ) -> Tensor:
        view = (slice(None), slice(None), slice(tile.y_start, tile.y_end), slice(tile.x_start, tile.x_end))
        left_tile = left[view]
        right_tile = right[view]
        mono_left_tile = mono_left[view] if mono_left is not None else None
        mono_right_tile = mono_right[view] if mono_right is not None else None

        # Pad to multiple of 32 like test.py does (lines 204-207)
        ht, wt = left_tile.shape[-2:]
        pad_ht = (((ht // 32) + 1) * 32 - ht) % 32
        pad_wd = (((wt // 32) + 1) * 32 - wt) % 32
        _pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        
        left_tile = torch.nn.functional.pad(left_tile, _pad, mode='replicate')
        right_tile = torch.nn.functional.pad(right_tile, _pad, mode='replicate')
        if mono_left_tile is not None:
            mono_left_tile = torch.nn.functional.pad(mono_left_tile, _pad, mode='replicate')
        if mono_right_tile is not None:
            mono_right_tile = torch.nn.functional.pad(mono_right_tile, _pad, mode='replicate')

        with torch.amp.autocast('cuda', enabled=mixed_precision):
            disp = self.model(left_tile, right_tile, mono_left_tile, mono_right_tile, *args, **kwargs)
        disp = self._canonicalize_output(disp)
        
        # Unpad output like test.py (lines 234-237)
        hd, wd = disp.shape[-2:]
        c = [_pad[2], hd-_pad[3], _pad[0], wd-_pad[1]]
        disp = disp[..., c[0]:c[1], c[2]:c[3]]
        
        return disp.detach().to(self.device)

    def _process_tiles_batch(
        self,
        tiles: Sequence[TileSpec],
        left: Tensor,
        right: Tensor,
        mono_left: Optional[Tensor],
        mono_right: Optional[Tensor],
        args: Sequence,
        kwargs: dict,
        mixed_precision: bool,
    ) -> List[Tensor]:
        stacked_left: List[Tensor] = []
        stacked_right: List[Tensor] = []
        stacked_mono_left: List[Tensor] = []
        stacked_mono_right: List[Tensor] = []
        pad_infos: List[tuple] = []

        view_fn = lambda t: (
            slice(None),
            slice(None),
            slice(t.y_start, t.y_end),
            slice(t.x_start, t.x_end),
        )

        for tile in tiles:
            view = view_fn(tile)
            left_tile = left[view]
            right_tile = right[view]
            
            # Pad each tile to multiple of 32
            ht, wt = left_tile.shape[-2:]
            pad_ht = (((ht // 32) + 1) * 32 - ht) % 32
            pad_wd = (((wt // 32) + 1) * 32 - wt) % 32
            _pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
            pad_infos.append(_pad)
            
            left_tile = torch.nn.functional.pad(left_tile, _pad, mode='replicate')
            right_tile = torch.nn.functional.pad(right_tile, _pad, mode='replicate')
            
            stacked_left.append(left_tile)
            stacked_right.append(right_tile)
            
            if mono_left is not None:
                mono_left_tile = mono_left[view]
                mono_left_tile = torch.nn.functional.pad(mono_left_tile, _pad, mode='replicate')
                stacked_mono_left.append(mono_left_tile)
            if mono_right is not None:
                mono_right_tile = mono_right[view]
                mono_right_tile = torch.nn.functional.pad(mono_right_tile, _pad, mode='replicate')
                stacked_mono_right.append(mono_right_tile)

        left_batch = torch.cat(stacked_left, dim=0)
        right_batch = torch.cat(stacked_right, dim=0)
        mono_left_batch = torch.cat(stacked_mono_left, dim=0) if stacked_mono_left else None
        mono_right_batch = torch.cat(stacked_mono_right, dim=0) if stacked_mono_right else None

        with torch.amp.autocast('cuda', enabled=mixed_precision):
            disp_batch = self.model(
                left_batch,
                right_batch,
                mono_left_batch,
                mono_right_batch,
                *args,
                **kwargs,
            )

        disp_batch = self._canonicalize_output(disp_batch)
        
        # Unpad each tile's output
        outputs = []
        for i, _pad in enumerate(pad_infos):
            disp = disp_batch[i : i + 1]
            hd, wd = disp.shape[-2:]
            c = [_pad[2], hd-_pad[3], _pad[0], wd-_pad[1]]
            disp = disp[..., c[0]:c[1], c[2]:c[3]]
            outputs.append(disp.detach().to(self.device))
        
        return outputs

    def _accumulate_tile(
        self,
        tile: TileSpec,
        disp: Tensor,
        stitched: Tensor,
        weight_map: Tensor,
    ) -> None:
        if disp.dim() != 4:
            raise ValueError("Model output must be BCHW tensor")
        if disp.shape[-2:] != (tile.height, tile.width):
            raise ValueError("Tile output spatial size mismatch")

        view = (slice(None), slice(None), slice(tile.y_start, tile.y_end), slice(tile.x_start, tile.x_end))
        weight = _make_blend_weight(tile.height, tile.width, disp.device)
        weight = weight.unsqueeze(0).unsqueeze(0)

        # Áp dụng global guidance nếu có
        if hasattr(self, '_global_guidance') and self._global_guidance is not None and self._guidance_weight > 0:
            # Trích xuất guidance cho tile hiện tại
            guidance_tile = self._global_guidance[view]
            
            # Tính toán confidence dựa trên sự khác biệt giữa guidance và prediction
            diff = torch.abs(disp - guidance_tile)
            max_diff = torch.max(diff) + 1e-6
            confidence = 1.0 - (diff / max_diff)
            
            # Blend giữa guidance và prediction dựa trên confidence
            guidance_influence = self._guidance_weight * confidence
            disp_blended = (1.0 - guidance_influence) * disp + guidance_influence * guidance_tile
            
            stitched[view] += disp_blended * weight
        else:
            stitched[view] += disp * weight
            
        weight_map[view] += weight
