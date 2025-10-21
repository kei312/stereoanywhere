"""CPU offloading utilities for StereoAnywhere inference."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import torch
import torch.nn as nn

Tensor = torch.Tensor


@contextmanager
def temporarily_to(module: nn.Module, device: torch.device) -> Iterator[nn.Module]:
    """Temporarily move a module to a device and restore afterwards."""

    original_device = next(module.parameters()).device
    if original_device != device:
        module.to(device)
    try:
        yield module
    finally:
        if original_device != device:
            module.to(original_device)
            torch.cuda.empty_cache()


class CPUOffloadWrapper(nn.Module):
    """Pipeline wrapper that offloads intermediate blocks to the CPU."""

    def __init__(
        self,
        model: nn.Module,
        mono_model: Optional[nn.Module] = None,
        offload_feature: bool = True,
        offload_mono: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.mono_model = mono_model
        self.offload_feature = offload_feature
        self.offload_mono = offload_mono

    @staticmethod
    def _detach_to_cpu(tensor: Optional[Tensor]) -> Optional[Tensor]:
        if tensor is None:
            return None
        return tensor.detach().to("cpu")

    def forward(
        self,
        left: Tensor,
        right: Tensor,
        mono_left: Optional[Tensor] = None,
        mono_right: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        device = left.device

        # Stage 1 – run monocular model if needed.
        if mono_left is None or mono_right is None:
            if self.mono_model is None:
                raise ValueError("Monocular model required when mono inputs absent")
            with temporarily_to(self.mono_model, device):
                with torch.cuda.amp.autocast(enabled=kwargs.get("mixed_precision", False)):
                    mono_left, mono_right = self.mono_model(left, right)
            if self.offload_mono:
                mono_left = self._detach_to_cpu(mono_left)
                mono_right = self._detach_to_cpu(mono_right)

        # Stage 2 – run StereoAnywhere on tiles.
        with temporarily_to(self.model, device):
            if mono_left is not None and mono_left.device != device:
                mono_left = mono_left.to(device)
            if mono_right is not None and mono_right.device != device:
                mono_right = mono_right.to(device)
            with torch.cuda.amp.autocast(enabled=kwargs.get("mixed_precision", False)):
                disparity = self.model(left, right, mono_left, mono_right, *args, **kwargs)

        # Free transient GPU allocations.
        torch.cuda.empty_cache()
        return disparity
