"""Helpers to mitigate non-Lambertian failure cases during tiled inference."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from .tiled_inference import MapReduceInference


@dataclass
class NonLambertianOutputs:
    disparity: np.ndarray
    mask: np.ndarray


class NonLambertianProcessor(MapReduceInference):
    """Extension of MapReduceInference with mirror-aware heuristics."""

    def __init__(
        self,
        *args,
        mirror_conf_th: float = 0.95,
        mirror_attenuation: float = 0.85,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mirror_conf_th = mirror_conf_th
        self.mirror_attenuation = mirror_attenuation

    @staticmethod
    def detect_nonlambertian(left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(left_gray, right_gray)
        _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def infer(
        self,
        left_img: np.ndarray,
        right_img: np.ndarray,
        *,
        verbose: bool = False,
        **kwargs,
    ) -> NonLambertianOutputs:
        mask = self.detect_nonlambertian(left_img, right_img)
        kwargs.setdefault("use_truncate_vol", True)
        kwargs.setdefault("mirror_conf_th", self.mirror_conf_th)
        kwargs.setdefault("mirror_attenuation", self.mirror_attenuation)

        disp = super().infer(left_img, right_img, verbose=verbose, **kwargs)
        return NonLambertianOutputs(disparity=disp, mask=mask)
