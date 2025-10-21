"""High level tiled inference pipeline for StereoAnywhere."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .tile_wrapper import TileWrapper
from .memory_utils import log_memory_snapshot, select_tiling_parameters

Tensor = torch.Tensor


@dataclass
class TiledInputs:
    left: Tensor
    right: Tensor
    mono_left: Optional[Tensor]
    mono_right: Optional[Tensor]


class MapReduceInference:
    """Orchestrate tiled inference for StereoAnywhere."""

    def __init__(
        self,
        stereo_model: torch.nn.Module,
        mono_model: Optional[Callable[[Tensor, Tuple[int, int]], Tensor]] = None,
        tile_size: Optional[int] = None,
        tile_width: Optional[int] = None,
        tile_height: Optional[int] = None,
        overlap: Optional[int] = None,
        batch_tiles: bool = False,
        mixed_precision: bool = False,
        clear_cache: bool = False,
        auto_tiling: bool = True,
        use_global_guidance: bool = False,
        guidance_scale: float = 2.0,
        guidance_weight: float = 0.3,
    ) -> None:
        self.stereo_model = stereo_model
        self.mono_model = mono_model
        self.mixed_precision = mixed_precision
        self.clear_cache = clear_cache
        self.use_global_guidance = use_global_guidance
        self.guidance_scale = guidance_scale
        self.guidance_weight = guidance_weight
        self._guidance_cache = {}  # Cache guidance maps

        # Xác định sử dụng tile hình chữ nhật hay hình vuông
        use_rectangular = tile_width is not None and tile_height is not None and tile_width > 0 and tile_height > 0
        
        if use_rectangular:
            # Sử dụng tile hình chữ nhật
            # StereoAnywhere downsamples by 32×, so keep tiling grid aligned to that stride.
            tile_width = int(max(32, (tile_width + 31) // 32 * 32))
            tile_height = int(max(32, (tile_height + 31) // 32 * 32))
            
            if overlap is None:
                params = select_tiling_parameters()
                overlap = params.overlap
            
            if overlap:
                # Overlap không nên lớn hơn kích thước nhỏ nhất của tile
                max_overlap = min(tile_width, tile_height) - 32
                overlap = int(min(max_overlap, (overlap + 31) // 32 * 32))
            else:
                overlap = 0
                
            self.tile_wrapper = TileWrapper(
                stereo_model,
                tile_width=tile_width,
                tile_height=tile_height,
                overlap=overlap,
                batch_tiles=batch_tiles,
            )
        else:
            # Sử dụng tile hình vuông (logic cũ)
            if tile_size is None or overlap is None:
                params = select_tiling_parameters()
                tile_size = params.tile_size
                overlap = params.overlap

            # StereoAnywhere downsamples by 32×, so keep tiling grid aligned to that stride.
            tile_size = int(max(32, (tile_size + 31) // 32 * 32))
            if overlap:
                overlap = int(min(tile_size - 32, (overlap + 31) // 32 * 32))
            else:
                overlap = 0
            self.tile_wrapper = TileWrapper(
                stereo_model,
                tile_size=tile_size,
                overlap=overlap,
                batch_tiles=batch_tiles,
            )

    def _prepare_inputs(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        iscale: float,
        mono_size: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
        mono_pair: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> TiledInputs:
        target_shape = (
            round(left_img.shape[0] / iscale),
            round(left_img.shape[1] / iscale),
        )

        left_tensor = torch.from_numpy(left_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        right_tensor = torch.from_numpy(right_img).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        if iscale != 1.0:
            left_resized = F.interpolate(left_tensor, size=target_shape, mode="bilinear", align_corners=False)
            right_resized = F.interpolate(right_tensor, size=target_shape, mode="bilinear", align_corners=False)
        else:
            left_resized = left_tensor
            right_resized = right_tensor

        left_resized = left_resized.to(device=device, dtype=dtype)
        right_resized = right_resized.to(device=device, dtype=dtype)

        mono_left = mono_right = None
        if mono_pair is not None:
            mono_left, mono_right = mono_pair
            mono_left = F.interpolate(mono_left, size=target_shape, mode="bilinear", align_corners=False)
            mono_right = F.interpolate(mono_right, size=target_shape, mode="bilinear", align_corners=False)
            mono_left = mono_left.to(device=device, dtype=dtype)
            mono_right = mono_right.to(device=device, dtype=dtype)
        elif self.mono_model is not None:
            mono_left = self.mono_model(left_tensor.to(device=device, dtype=dtype), mono_size)
            mono_right = self.mono_model(right_tensor.to(device=device, dtype=dtype), mono_size)
            mono_left = F.interpolate(mono_left, size=target_shape, mode="bilinear", align_corners=False)
            mono_right = F.interpolate(mono_right, size=target_shape, mode="bilinear", align_corners=False)
            mono_left = mono_left.to(device=device, dtype=dtype)
            mono_right = mono_right.to(device=device, dtype=dtype)

        return TiledInputs(left=left_resized, right=right_resized, mono_left=mono_left, mono_right=mono_right)

    def _compute_global_guidance(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        mono_pair: Optional[Tuple[Tensor, Tensor]],
        device: torch.device,
        dtype: torch.dtype,
        verbose: bool = False,
    ) -> Optional[np.ndarray]:
        """Tính toán global guidance từ low-res inference."""
        if not self.use_global_guidance:
            return None
        
        # Tạo cache key từ image content
        import hashlib
        cache_key = hashlib.md5(left_img.tobytes()).hexdigest()
        
        # Check cache
        if cache_key in self._guidance_cache:
            if verbose:
                print(f"[Guidance] Using cached guidance")
            return self._guidance_cache[cache_key]
        
        if verbose:
            print(f"[Guidance] Computing global guidance at scale {self.guidance_scale}...")
        
        # Downscale images
        h, w = left_img.shape[:2]
        target_h = int(h / self.guidance_scale)
        target_w = int(w / self.guidance_scale)
        
        import cv2
        left_lowres = cv2.resize(left_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        right_lowres = cv2.resize(right_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # Prepare tensors
        left_tensor = torch.from_numpy(left_lowres).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        right_tensor = torch.from_numpy(right_lowres).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        left_tensor = left_tensor.to(device=device, dtype=dtype)
        right_tensor = right_tensor.to(device=device, dtype=dtype)
        
        # Prepare mono for low-res
        mono_left_lr = mono_right_lr = None
        if mono_pair is not None:
            mono_left_lr = F.interpolate(mono_pair[0], size=(target_h, target_w), mode='bilinear', align_corners=False)
            mono_right_lr = F.interpolate(mono_pair[1], size=(target_h, target_w), mode='bilinear', align_corners=False)
        elif self.mono_model is not None:
            mono_left_lr = self.mono_model(left_tensor, (target_h, target_w))
            mono_right_lr = self.mono_model(right_tensor, (target_h, target_w))
        
        # Run inference at low-res
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=self.mixed_precision):
                output = self.stereo_model(
                    left_tensor,
                    right_tensor,
                    mono_left_lr,
                    mono_right_lr,
                    iters=32,
                    test_mode=True
                )
        
        # Extract disparity
        if isinstance(output, (tuple, list)):
            disp_lowres = output[0]
        else:
            disp_lowres = output
        
        if disp_lowres.dim() == 3:
            disp_lowres = disp_lowres.unsqueeze(1)
        
        disp_lowres = -disp_lowres.squeeze(0).squeeze(0).float().cpu().numpy()
        
        # Upscale to original size
        disp_guidance = cv2.resize(disp_lowres, (w, h), interpolation=cv2.INTER_LINEAR)
        disp_guidance *= self.guidance_scale  # Scale disparity values
        
        # Cache result
        self._guidance_cache[cache_key] = disp_guidance
        
        if verbose:
            print(f"[Guidance] Computed and cached guidance: {disp_guidance.shape}")
        
        return disp_guidance

    def infer(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        *,
        iscale: float = 1.0,
        oscale: float = 1.0,
        mono_size: Tuple[int, int] = (518, 518),
        post_scale: float = 1.0,
        verbose: bool = False,
        mono_pair: Optional[Tuple[Tensor, Tensor]] = None,
        global_guidance: Optional[np.ndarray] = None,
        guidance_weight: float = 0.3,
        **kwargs,
    ) -> np.ndarray:
        device = next(self.stereo_model.parameters()).device
        dtype = next(self.stereo_model.parameters()).dtype
        inputs = self._prepare_inputs(
            left_img,
            right_img,
            iscale,
            mono_size,
            device,
            dtype,
            mono_pair=mono_pair,
        )

        # Tự động tính toán hoặc sử dụng global guidance
        guidance_tensor = None
        
        # Nếu bật use_global_guidance và chưa có guidance, tự động tính
        if self.use_global_guidance and global_guidance is None:
            global_guidance = self._compute_global_guidance(
                left_img,
                right_img,
                mono_pair,
                device,
                dtype,
                verbose=verbose
            )
        
        # Xử lý global guidance nếu có
        if global_guidance is not None:
            # Chuyển đổi guidance sang tensor
            if isinstance(global_guidance, np.ndarray):
                guidance_tensor = torch.from_numpy(global_guidance).float()
            else:
                guidance_tensor = global_guidance.float()
            
            # Resize guidance để khớp với kích thước input
            target_shape = inputs.left.shape[-2:]
            if guidance_tensor.shape[-2:] != target_shape:
                guidance_tensor = guidance_tensor.unsqueeze(0).unsqueeze(0) if guidance_tensor.dim() == 2 else guidance_tensor
                guidance_tensor = F.interpolate(
                    guidance_tensor,
                    size=target_shape,
                    mode='bilinear',
                    align_corners=False
                )
                # Điều chỉnh giá trị disparity theo tỷ lệ resize
                scale_factor = target_shape[1] / guidance_tensor.shape[-1]
                guidance_tensor = guidance_tensor * scale_factor
            
            guidance_tensor = guidance_tensor.to(device=device, dtype=dtype)
            
            # Sử dụng guidance_weight từ instance nếu không được truyền vào
            if guidance_weight == 0.3:  # Default value
                guidance_weight = self.guidance_weight
            
            if verbose:
                print(f"Global guidance loaded: {guidance_tensor.shape}, weight={guidance_weight}")

        if verbose:
            log_memory_snapshot(prefix="[MapReduce] before forward • ")

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=self.mixed_precision):
                disp = self.tile_wrapper(
                    inputs.left,
                    inputs.right,
                    inputs.mono_left,
                    inputs.mono_right,
                    mixed_precision=self.mixed_precision,
                    global_guidance=guidance_tensor,
                    guidance_weight=guidance_weight,
                    **kwargs,
                )
        disp = disp.squeeze(0).squeeze(0).float().cpu().numpy()

        if oscale != iscale or post_scale != 1.0:
            target = (
                round(left_img.shape[0] / oscale),
                round(left_img.shape[1] / oscale),
            )
            disp_tensor = torch.from_numpy(disp).unsqueeze(0).unsqueeze(0)
            disp_tensor = F.interpolate(disp_tensor, size=target, mode="nearest")
            disp = disp_tensor.squeeze().numpy() * (iscale / oscale) * post_scale
        else:
            disp *= post_scale

        if self.clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        if verbose:
            log_memory_snapshot(prefix="[MapReduce] after forward • ")

        return disp
