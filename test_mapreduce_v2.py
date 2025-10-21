"""Evaluation script for StereoAnywhere using MapReduce tiling."""
from __future__ import annotations

import argparse
import gc
import math
import os
import random
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from torch import autocast

from dataloaders import fetch_dataloader
from losses import guided_metrics
from mapreduce_v2 import MapReduceInference, NonLambertianProcessor, select_tiling_parameters
from models.depth_anything_v2 import get_depth_anything_v2
from models.stereoanywhere import StereoAnywhere
from utils import color_error_image_kitti, guided_visualize

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="StereoAnywhere MapReduce evaluation")
    parser.add_argument("--datapath", required=True)
    parser.add_argument("--dataset", default="middlebury")
    parser.add_argument("--loadstereomodel", required=True)
    parser.add_argument("--loadmonomodel", required=True)
    parser.add_argument("--stereomodel", default="stereoanywhere")
    parser.add_argument("--monomodel", default="DAv2")
    parser.add_argument("--vit_encoder", default="vitl", choices=["vitl", "vitb", "vits"])
    parser.add_argument("--maxdisp", type=int, default=192)
    parser.add_argument("--iscale", type=float, default=1.0)
    parser.add_argument("--oscale", type=float, default=1.0)
    parser.add_argument("--tries", type=int, default=1)
    parser.add_argument("--valsize", type=int, default=0)
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--errormetric", default="bad 3.0")
    parser.add_argument("--dilation", type=int, default=1)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--numworkers", type=int, default=1)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--iters", type=int, default=32)
    parser.add_argument("--vol_n_masks", type=int, default=8)
    parser.add_argument("--vol_downsample", type=float, default=0)
    parser.add_argument("--use_truncate_vol", action="store_true")
    parser.add_argument("--use_aggregate_mono_vol", action="store_true")
    parser.add_argument("--use_aggregate_stereo_vol", action="store_true")
    parser.add_argument("--mirror_conf_th", type=float, default=0.95)
    parser.add_argument("--mirror_attenuation", type=float, default=0.85)
    ## THAM SỐ TILING
    parser.add_argument("--overfit", action="store_true", default=False)
    parser.add_argument("--tile_preset", type=str, default=None, 
                        help="Sử dụng preset có sẵn (ví dụ: 'middlebury', 'kitti', 'default'). "
                             "Dùng 'list' để xem tất cả preset. "
                             "Dùng 'auto' để tự động chọn dựa trên dataset.")
    parser.add_argument("--tile_size", type=int, default=0, help="Tile size (0 = auto, for square tiles)")
    parser.add_argument("--tile_width", type=int, default=0, help="Tile width (0 = use tile_size, overrides tile_size if > 0)")
    parser.add_argument("--tile_height", type=int, default=0, help="Tile height (0 = use tile_size, overrides tile_size if > 0)")
    parser.add_argument("--overlap", type=int, default=0, help="Tile overlap (0 = auto)")
    parser.add_argument("--batch_tiles", action="store_true")
    parser.add_argument("--clear_cache", action="store_true")
    parser.add_argument("--non_lambertian", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    ## GLOBAL GUIDANCE
    parser.add_argument("--use_global_guidance", action="store_true",
                        help="Enable automatic global guidance computation from low-res pass")
    parser.add_argument("--guidance_scale", type=float, default=2.0,
                        help="Scale factor for global guidance low-res pass (default 2.0)")
    parser.add_argument("--guidance_weight", type=float, default=0.3,
                        help="Weight of global guidance influence (0-1, default 0.3)")
    parser.add_argument("--guidance_dir", type=str, default=None,
                        help="[DEPRECATED] Use --use_global_guidance instead")
    return parser


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_models(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> Dict[str, Optional[torch.nn.Module]]:
    stereo = StereoAnywhere(args)
    stereo = nn.DataParallel(stereo)
    state = torch.load(args.loadstereomodel, map_location="cpu")
    state = state.get("state_dict", state)
    stereo.load_state_dict(state, strict=True)
    stereo = stereo.module.to(device=device, dtype=dtype).eval()

    mono = None
    if args.monomodel == "DAv2":
        mono = get_depth_anything_v2(args.loadmonomodel, encoder=args.vit_encoder, map_location=device)
        mono = mono.to(device=device, dtype=dtype).eval()

    return {"stereo": stereo, "mono": mono}


def compute_mono_pair(
    im2: torch.Tensor,
    im3: torch.Tensor,
    dataset: str,
    mono_model: Optional[nn.Module],
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if mono_model is None:
        return None

    width_dict = {
        "kitti2012": 1372,
        "kitti2015": 1372,
        "eth3d": 518,
        "middlebury": 518 * 2,
        "middlebury2021": 1372,
        "booster": 518 * 2,
        "layeredflow": 952,
        "monkaa": 960,
    }
    height_dict = {
        "kitti2012": 518,
        "kitti2015": 518,
        "eth3d": 518,
        "middlebury": 518 * 2,
        "middlebury2021": 770,
        "booster": 756,
        "layeredflow": 532,
        "monkaa": 544,
    }
    width = width_dict.get(dataset, 518)
    height = height_dict.get(dataset, 518)

    # DepthAnything expects spatial dims divisible by 14 (ViT patch size).
    width = max(width, im2.shape[-1])
    height = max(height, im2.shape[-2])
    width = int(math.ceil(width / 14.0) * 14)
    height = int(math.ceil(height / 14.0) * 14)

    stacked = torch.cat([im2, im3], dim=0).to(device=device, dtype=dtype)
    with torch.no_grad():
        mono_depths = mono_model.infer_image(
            stacked, input_size_width=width, input_size_height=height
        )
    mono_depths = mono_depths.to(device=device, dtype=dtype)
    mono_depths = (mono_depths - mono_depths.min()) / (mono_depths.max() - mono_depths.min() + 1e-8)
    return mono_depths


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    # More memory-efficient conversion
    tensor_cpu = tensor.detach().cpu()
    tensor_cpu = tensor_cpu.squeeze(0)
    tensor_cpu = tensor_cpu.permute(1, 2, 0)
    image = tensor_cpu.numpy()
    image = np.clip(image, 0.0, 1.0)
    result = (image * 255).astype(np.uint8)
    
    # Clear intermediate tensors
    del tensor_cpu, image
    
    return result


def run_mapreduce(
        
    datablob: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    inferencer: MapReduceInference,
    mono_model: Optional[nn.Module],
) -> Dict[str, torch.Tensor]:
    data = {k: v.clone() for k, v in datablob.items() if torch.is_tensor(v)}

    if "maskocc" not in data:
        data["maskocc"] = torch.zeros_like(data["gt"])

    # Apply input/output scaling like test.py
    if args.iscale != 1:
        data["im2"] = F.interpolate(data["im2"], scale_factor=1.0 / args.iscale, mode='bilinear', align_corners=False)
        data["im3"] = F.interpolate(data["im3"], scale_factor=1.0 / args.iscale, mode='bilinear', align_corners=False)

    if args.oscale != 1:
        data["gt"] = F.interpolate(data["gt"], scale_factor=1.0 / args.oscale, mode="nearest") / args.oscale
        data["validgt"] = F.interpolate(data["validgt"], scale_factor=1.0 / args.oscale, mode="nearest")
        data["maskocc"] = F.interpolate(data["maskocc"], scale_factor=1.0 / args.oscale, mode="nearest")

    im2 = data["im2"].to(device=device, dtype=dtype)
    im3 = data["im3"].to(device=device, dtype=dtype)

    # Compute monocular depth
    mono_depths = compute_mono_pair(im2, im3, args.dataset, mono_model, dtype, device)
    mono_pair = None
    if mono_depths is not None:
        mono_left = mono_depths[0:1].detach()
        mono_right = mono_depths[1:2].detach()
        mono_pair = (mono_left, mono_right)
    else:
        mono_left = torch.zeros_like(im2[:, :1])
        mono_right = torch.zeros_like(im3[:, :1])
        mono_pair = (mono_left, mono_right)

    # Pad to multiple of 32 (like test.py does)
    ht, wt = im2.shape[-2], im2.shape[-1]
    pad_ht = (((ht // 32) + 1) * 32 - ht) % 32
    pad_wd = (((wt // 32) + 1) * 32 - wt) % 32
    _pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
    
    im2_padded = F.pad(im2, _pad, mode='replicate')
    im3_padded = F.pad(im3, _pad, mode='replicate')
    mono_left_padded = F.pad(mono_left, _pad, mode='replicate')
    mono_right_padded = F.pad(mono_right, _pad, mode='replicate')
    mono_pair_padded = (mono_left_padded, mono_right_padded)

    left_np = tensor_to_numpy_image(im2_padded)
    right_np = tensor_to_numpy_image(im3_padded)

    # Model sẽ tự động tính toán guidance nếu use_global_guidance=True
    # Không cần load guidance thủ công nữa
    disp = inferencer.infer(
        left_np,
        right_np,
        iscale=1.0,  # Already scaled above
        oscale=1.0,  # Already scaled above
        mono_size=im2_padded.shape[-2:],
        verbose=args.verbose,
        mono_pair=mono_pair_padded,
        iters=args.iters,
        test_mode=True,
    )

    gt = data["gt"].detach().cpu()
    valid = data["validgt"].detach().cpu()
    maskocc = data["maskocc"].detach().cpu()

    # Convert disparity to tensor and ensure proper shape (BCHW format)
    disp_tensor = torch.from_numpy(disp).float()
    if disp_tensor.dim() == 2:
        # HW -> BCHW
        disp_tensor = disp_tensor.unsqueeze(0).unsqueeze(0)
    elif disp_tensor.dim() == 3:
        # BHW -> BCHW
        disp_tensor = disp_tensor.unsqueeze(1)
    
    # Remove padding (like test.py does)
    hd, wd = disp_tensor.shape[-2:]
    c = [_pad[2], hd-_pad[3], _pad[0], wd-_pad[1]]
    disp_tensor = disp_tensor[..., c[0]:c[1], c[2]:c[3]]
    
    # Resize prediction to match ground truth dimensions if needed
    if args.iscale != 1 and args.iscale / args.oscale != 1:
        disp_tensor = F.interpolate(
            disp_tensor,
            size=gt.shape[-2:],
            mode='nearest'  # Use nearest like test.py
        ) * args.iscale / args.oscale
    elif disp_tensor.shape[-2:] != gt.shape[-2:]:
        # Fallback resize
        orig_h, orig_w = disp_tensor.shape[-2:]
        target_h, target_w = gt.shape[-2:]
        
        disp_tensor = F.interpolate(
            disp_tensor,
            size=(target_h, target_w),
            mode='nearest'
        )
        
        # Scale disparity values proportionally (use width ratio for horizontal disparity)
        scale_factor = target_w / orig_w
        disp_tensor = disp_tensor * scale_factor

    metrics = guided_metrics(
        disp_tensor.numpy(),
        gt.numpy(),
        valid.numpy(),
        maskocc.numpy(),
    )

    metrics["disp"] = disp_tensor
    metrics["im2"] = im2.detach().cpu()
    metrics["im3"] = im3.detach().cpu()
    metrics["mono_left"] = mono_pair[0].detach().cpu()
    metrics["mono_right"] = mono_pair[1].detach().cpu()
    metrics["gt"] = gt
    metrics["validgt"] = valid
    metrics["maskocc"] = maskocc
    return metrics


def ensure_outdirs(base: Path) -> None:
    for dirname in ["dmap", "left", "right", "gt", "maemap", "metricmap", "mono_left", "mono_right", "raw"]:
        (base / dirname).mkdir(parents=True, exist_ok=True)


def save_visualizations(
    output_dir: Path,
    idx: int,
    results: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> None:
    ensure_outdirs(output_dir)

    left_img = tensor_to_numpy_image(results["im2"])
    right_img = tensor_to_numpy_image(results["im3"])

    cv2.imwrite(str(output_dir / "left" / f"{idx}.png"), cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "right" / f"{idx}.png"), cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR))

    disp = results["disp"].squeeze(0)
    max_val = torch.where(torch.isinf(results["gt"][0]), -float("inf"), results["gt"][0]).max()
    max_val = disp.max() if max_val == 0 else max_val

    gt_color = cv2.applyColorMap(
        (torch.clamp(results["gt"][0, 0], 0, max_val) / max_val * 255).numpy().astype(np.uint8),
        cv2.COLORMAP_VIRIDIS,
    )
    cv2.imwrite(str(output_dir / "gt" / f"{idx}.png"), gt_color)

    disp_color = cv2.applyColorMap(
        ((torch.clamp(disp, 0, max_val) / max_val) * 255).numpy().astype(np.uint8),
        cv2.COLORMAP_INFERNO,
    )
    cv2.imwrite(str(output_dir / "dmap" / f"{idx}.png"), disp_color)

    errormap = color_error_image_kitti(
        torch.abs(results["gt"] - disp.unsqueeze(0)).squeeze().numpy(),
        scale=1,
        mask=results["gt"].numpy() > 0,
        dilation=args.dilation,
    )
    cv2.imwrite(str(output_dir / "maemap" / f"{idx}.png"), errormap)

    metricmap = guided_visualize(
        disp.numpy(),
        results["gt"].squeeze().numpy(),
        results["gt"].squeeze().numpy() > 0,
        dilation=args.dilation,
    )[args.errormetric]
    cv2.imwrite(str(output_dir / "metricmap" / f"{idx}.png"), metricmap)

    mono_left = (results["mono_left"] * 255).squeeze().numpy().astype(np.uint8)
    mono_right = (results["mono_right"] * 255).squeeze().numpy().astype(np.uint8)
    cv2.imwrite(str(output_dir / "mono_left" / f"{idx}.png"), mono_left)
    cv2.imwrite(str(output_dir / "mono_right" / f"{idx}.png"), mono_right)

    np.save(str(output_dir / "raw" / f"{idx:06}_10.npy"), disp.numpy())


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Xử lý tile preset nếu được chỉ định
    if args.tile_preset:
        from mapreduce_v2 import list_presets, get_preset, get_preset_for_dataset
        
        if args.tile_preset.lower() == 'list':
            list_presets()
            return
        
        # Lấy preset
        if args.tile_preset.lower() == 'auto':
            preset = get_preset_for_dataset(args.dataset)
        else:
            preset = get_preset(args.tile_preset)
        
        # Áp dụng preset (chỉ nếu chưa được set thủ công)
        if args.tile_width <= 0:
            args.tile_width = preset.tile_width
        if args.tile_height <= 0:
            args.tile_height = preset.tile_height
        if args.overlap <= 0:
            args.overlap = preset.overlap
        
        print(f"Using preset '{preset.name}': {preset.tile_width}x{preset.tile_height}, overlap={preset.overlap}")
        print(f"Description: {preset.description}")

    assert args.iscale > 0
    assert args.oscale > 0

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    dtype = torch.float16 if args.half else torch.float32

    set_random_seeds(args.seed)

    models = load_models(args, device, dtype)
    stereonet = models["stereo"]
    mono_model = models["mono"]

    # Xác định xem có sử dụng tile hình chữ nhật hay không
    use_rectangular_tiles = args.tile_width > 0 and args.tile_height > 0
    
    # Xác định có sử dụng global guidance hay không
    use_guidance = args.use_global_guidance
    guidance_scale = getattr(args, 'guidance_scale', 2.0)
    guidance_weight = getattr(args, 'guidance_weight', 0.3)
    
    if use_rectangular_tiles:
        # Sử dụng tile hình chữ nhật với width và height riêng biệt
        tile_width = args.tile_width
        tile_height = args.tile_height
        
        if args.overlap <= 0:
            params = select_tiling_parameters()
            overlap = params.overlap
        else:
            overlap = args.overlap
            
        print(f"Rectangular tiling: tile_width={tile_width}, tile_height={tile_height}, overlap={overlap}")
        if use_guidance:
            print(f"Global guidance: enabled (scale={guidance_scale}, weight={guidance_weight})")
        
        inference_cls = NonLambertianProcessor if args.non_lambertian else MapReduceInference
        inferencer = inference_cls(
            stereonet,
            mono_model=None,
            tile_width=tile_width,
            tile_height=tile_height,
            overlap=overlap,
            batch_tiles=args.batch_tiles,
            mixed_precision=args.mixed_precision,
            clear_cache=args.clear_cache,
            use_global_guidance=use_guidance,
            guidance_scale=guidance_scale,
            guidance_weight=guidance_weight,
        )
    else:
        # Sử dụng tile hình vuông (logic cũ)
        if args.tile_size <= 0 or args.overlap <= 0:
            params = select_tiling_parameters()
            tile_size, overlap = params.tile_size, params.overlap
            print(f"Auto-selected square tiling: tile_size={tile_size}, overlap={overlap}")
        else:
            tile_size, overlap = args.tile_size, args.overlap
            print(f"Manual square tiling: tile_size={tile_size}, overlap={overlap}")
        
        if use_guidance:
            print(f"Global guidance: enabled (scale={guidance_scale}, weight={guidance_weight})")

        inference_cls = NonLambertianProcessor if args.non_lambertian else MapReduceInference
        inferencer = inference_cls(
            stereonet,
            mono_model=None,
            tile_size=tile_size,
            overlap=overlap,
            batch_tiles=args.batch_tiles,
            mixed_precision=args.mixed_precision,
            clear_cache=args.clear_cache,
            use_global_guidance=use_guidance,
            guidance_scale=guidance_scale,
            guidance_weight=guidance_weight,
        )

    args.test = True
    args.batch_size = 1
    loader = fetch_dataloader(args)
    print(f"Testing with {len(loader)} image pairs")

    output_dir = Path(args.outdir) if args.outdir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    acc_list: list[Dict[str, list]] = []
    for _ in range(args.tries):
        acc: Dict[str, list] = {}
        pbar = tqdm.tqdm(total=len(loader))
        val_len = min(len(loader), args.valsize) if args.valsize > 0 else len(loader)
        for idx, sample in enumerate(loader):
            if idx >= val_len:
                break

            result = run_mapreduce(sample, args, device, dtype, inferencer, mono_model)

            for key, value in result.items():
                if key in {"disp", "im2", "im3", "mono_left", "mono_right", "gt", "validgt", "maskocc"}:
                    continue
                # Handle both scalar and array values from guided_metrics
                if isinstance(value, np.ndarray):
                    if value.size > 1:
                        # Multi-element array: take mean
                        acc.setdefault(key, []).append(float(np.mean(value)))
                    else:
                        # Single-element array: extract scalar
                        acc.setdefault(key, []).append(float(value.item()))
                else:
                    # Already a scalar
                    acc.setdefault(key, []).append(float(value))

            if output_dir is not None:
                save_visualizations(output_dir, idx, result, args)

            # Explicit memory cleanup after each iteration to prevent OOM
            if 'disp' in result and torch.is_tensor(result['disp']):
                del result['disp']
            if 'mono_left' in result and torch.is_tensor(result['mono_left']):
                del result['mono_left']
            if 'mono_right' in result and torch.is_tensor(result['mono_right']):
                del result['mono_right']
            
            # Force garbage collection and clear CUDA cache
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Display comprehensive metrics in a readable format
            desc_parts = [f"[{idx+1}/{val_len}]"]
            
            # Show key bad metrics for progress bar
            for k in ["bad 2.0", "bad 4.0"]:
                if k in acc:
                    desc_parts.append(f"{k}:{np.mean(acc[k])*100:.2f}%")
            
            # Show error metrics
            for k in ["avgerr", "rms"]:
                if k in acc:
                    desc_parts.append(f"{k}:{np.mean(acc[k]):.3f}")
            
            pbar.set_description(" | ".join(desc_parts))
            pbar.update(1)
        pbar.close()

        acc_list.append(acc)

    acc_mean: Dict[str, float] = {}
    acc_std: Dict[str, float] = {}

    for acc in acc_list:
        for k, values in acc.items():
            arr = np.array(values, dtype=np.float32)
            acc_mean.setdefault(k, []).append(np.nanmean(arr))
            acc_std.setdefault(k, []).append(np.nanmean(arr))

    for k in acc_mean:
        acc_mean[k] = float(np.nanmean(acc_mean[k]))
        acc_std[k] = float(np.nanstd(acc_std[k]))

    print("MEAN Metrics:")

    # Print header exactly like test.py with all metrics
    metrs = ""
    # Define the exact order of metrics to match test.py output
    metric_order = [
        "bad 1.0", "bad 2.0", "bad 3.0", "bad 4.0", "bad 5.0", "bad 6.0", "bad 7.0", "bad 8.0",
        "avgerr", "rms",
        "occ bad 1.0", "occ bad 2.0", "occ bad 3.0", "occ bad 4.0", "occ bad 5.0", "occ bad 6.0", "occ bad 7.0", "occ bad 8.0",
        "occ avgerr", "occ rms",
        "noc bad 1.0", "noc bad 2.0", "noc bad 3.0", "noc bad 4.0", "noc bad 5.0", "noc bad 6.0", "noc bad 7.0", "noc bad 8.0",
        "noc avgerr", "noc rms"
    ]
    
    # Print header
    for k in metric_order:
        if k in acc_mean:
            metrs += f" {k.upper()} &"
    print(metrs)

    # Print values
    metrs = ""
    for k in metric_order:
        if k in acc_mean:
            if "bad" in k:
                metrs += f" {acc_mean[k] * 100:.2f} &"
            else:
                metrs += f" {acc_mean[k]:.2f} &"
    print(metrs)

    print("STD Metrics:")

    # Print STD values in same order
    metrs = ""
    for k in metric_order:
        if k in acc_std:
            if "bad" in k:
                metrs += f" {acc_std[k] * 100:.2f} &"
            else:
                metrs += f" {acc_std[k]:.2f} &"
    print(metrs)


if __name__ == "__main__":
    main()
