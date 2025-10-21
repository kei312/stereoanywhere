"""Fast demo script that runs StereoAnywhere with MapReduce tiling."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import autocast
from tqdm import tqdm

# Ensure project root on path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.stereoanywhere import StereoAnywhere  # type: ignore  # pylint: disable=wrong-import-position
from models.depth_anything_v2 import get_depth_anything_v2  # type: ignore
from mapreduce_v2 import MapReduceInference, NonLambertianProcessor, select_tiling_parameters


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StereoAnywhere tiled inference demo")
    parser.add_argument("--left", nargs="+", required=True, help="Left image(s)")
    parser.add_argument("--right", nargs="+", required=True, help="Right image(s)")
    parser.add_argument("--loadstereomodel", required=True, help="Path to StereoAnywhere weights")
    parser.add_argument("--loadmonomodel", required=True, help="Path to monocular model weights")
    parser.add_argument("--outdir", default=None, help="Directory to save outputs")
    parser.add_argument("--iscale", type=float, default=1.0, help="Input downscale factor")
    parser.add_argument("--oscale", type=float, default=1.0, help="Output scale factor")
    parser.add_argument("--mono_width", type=int, default=518, help="Mono model width")
    parser.add_argument("--mono_height", type=int, default=518, help="Mono model height")
    parser.add_argument("--tile_size", type=int, default=0, help="Tile size (0 = auto)")
    parser.add_argument("--overlap", type=int, default=0, help="Tile overlap (0 = auto)")
    parser.add_argument("--mixed_precision", action="store_true", help="Enable autocast")
    parser.add_argument("--half", action="store_true", help="Use FP16")
    parser.add_argument("--non_lambertian", action="store_true", help="Enable mirror-aware processing")
    parser.add_argument("--clear_cache", action="store_true", help="Clear CUDA cache between images")
    parser.add_argument("--verbose", action="store_true", help="Print memory diagnostics")
    parser.add_argument("--iters", type=int, default=32, help="StereoAnywhere iterations")
    parser.add_argument("--vol_n_masks", type=int, default=8)
    parser.add_argument("--use_truncate_vol", action="store_true")
    parser.add_argument("--use_aggregate_mono_vol", action="store_true")
    parser.add_argument("--use_aggregate_stereo_vol", action="store_true")
    parser.add_argument("--mirror_conf_th", type=float, default=0.95)
    parser.add_argument("--mirror_attenuation", type=float, default=0.85)
    return parser


def load_models(args: argparse.Namespace) -> Tuple[nn.Module, nn.Module]:
    stereo_args = argparse.Namespace(**vars(args))
    model = StereoAnywhere(stereo_args)
    model = nn.DataParallel(model)
    state = torch.load(args.loadstereomodel, map_location="cpu")
    state = state.get("state_dict", state)
    model.load_state_dict(state, strict=True)
    model = model.module.eval().cuda()

    mono_model = get_depth_anything_v2(args.loadmonomodel)
    mono_model.eval().cuda()
    return model, mono_model


def infer_pair(
    inferencer: MapReduceInference,
    left_path: Path,
    right_path: Path,
    args: argparse.Namespace,
) -> np.ndarray:
    left = cv2.cvtColor(cv2.imread(str(left_path)), cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(cv2.imread(str(right_path)), cv2.COLOR_BGR2RGB)
    disp = inferencer.infer(
        left,
        right,
        iscale=args.iscale,
        oscale=args.oscale,
        mono_size=(args.mono_height, args.mono_width),
        verbose=args.verbose,
        iters=args.iters,
        vol_n_masks=args.vol_n_masks,
        use_truncate_vol=args.use_truncate_vol,
        use_aggregate_mono_vol=args.use_aggregate_mono_vol,
        use_aggregate_stereo_vol=args.use_aggregate_stereo_vol,
    )
    return disp


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if len(args.left) != len(args.right):
        parser.error("--left and --right must have same length")

    if args.outdir is not None:
        os.makedirs(args.outdir, exist_ok=True)

    torch.set_grad_enabled(False)
    dtype = torch.float16 if args.half else torch.float32
    torch.set_float32_matmul_precision("high")

    stereonet, mono_model = load_models(args)
    stereonet = stereonet.to(dtype=dtype)
    mono_model = mono_model.to(dtype=dtype)

    if args.tile_size <= 0 or args.overlap <= 0:
        params = select_tiling_parameters()
        tile_size = params.tile_size
        overlap = params.overlap
    else:
        tile_size = args.tile_size
        overlap = args.overlap

    inference_cls = NonLambertianProcessor if args.non_lambertian else MapReduceInference
    inferencer = inference_cls(
        stereonet,
        mono_model=lambda tensor, size: mono_model.infer_image(
            tensor, input_size_width=size[1], input_size_height=size[0]
        ),
        tile_size=tile_size,
        overlap=overlap,
        mixed_precision=args.mixed_precision,
        clear_cache=args.clear_cache,
    )

    if args.verbose:
        print(f"[MapReduce] tile_size={tile_size}, overlap={overlap}")

    for left_path, right_path in tqdm(zip(args.left, args.right), total=len(args.left)):
        disparity = infer_pair(inferencer, Path(left_path), Path(right_path), args)

        if args.outdir:
            np.save(os.path.join(args.outdir, Path(left_path).stem + "_disp.npy"), disparity)
            color = cv2.applyColorMap(cv2.convertScaleAbs(disparity * 5, alpha=255), cv2.COLORMAP_INFERNO)
            cv2.imwrite(
                os.path.join(args.outdir, Path(left_path).stem + "_disp.png"),
                cv2.cvtColor(color, cv2.COLOR_RGB2BGR),
            )

    if args.verbose:
        print("Done")


if __name__ == "__main__":
    main()
