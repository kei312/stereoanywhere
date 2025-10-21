import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import tqdm
import os
from torch import autocast

# add the parent directory to the system path
import sys
import os
# Get the directory containing this script (demo/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project root)
project_root = os.path.dirname(script_dir)
# Add project root to path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Custom models
from models.stereoanywhere import StereoAnywhere as StereoAnywhere

# Monocular models
from models.depth_anything_v2 import get_depth_anything_v2

# MapReduce support
try:
    from mapreduce_v00xx import AdaptiveResolutionWrapper, TiledStereoWrapper, MemoryEfficientWrapper
    MAPREDUCE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  MapReduce module not available: {e}")
    MAPREDUCE_AVAILABLE = False

torch.set_float32_matmul_precision('high')

def main():
    parser = argparse.ArgumentParser(description='StereoAnywhere Fast Inference with MapReduce Support')

    parser.add_argument('--left', nargs='+', required=True, help='left image path(s)')
    parser.add_argument('--right', nargs='+', required=True, help='right image path(s)')

    parser.add_argument('--iscale', type=float, default=1.0, help='scale factor for input images')
    parser.add_argument('--outdir', default=None, type=str, help='output directory')
    parser.add_argument('--display_qualitatives', action='store_true', help='display qualitative results')
    parser.add_argument('--save_qualitatives', action='store_true', help='save qualitative results')

    parser.add_argument('--half', action='store_true', help='use half precision for inference')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision for inference')

    parser.add_argument('--stereomodel', default='stereoanywhere', help='select stereo model')
    parser.add_argument('--monomodel', default='DAv2', help='select mono model')

    parser.add_argument('--loadstereomodel', required=True, help='load stereo model')         
    parser.add_argument('--loadmonomodel', required=True, help='load mono model')

    parser.add_argument('--mono_width', type=int, default=518, help='Input width for the mono model')
    parser.add_argument('--mono_height', type=int, default=518, help='Input height for the mono model')
    parser.add_argument('--n_downsample', type=int, default=2)
    parser.add_argument('--n_additional_hourglass', type=int, default=0)
    parser.add_argument('--volume_channels', type=int, default=8)
    parser.add_argument('--vol_downsample', type=float, default=0)
    parser.add_argument('--vol_n_masks', type=int, default=8)
    parser.add_argument('--use_truncate_vol', action='store_true')
    parser.add_argument('--mirror_conf_th', type=float, default=0.98)
    parser.add_argument('--mirror_attenuation', type=float, default=0.9)
    parser.add_argument('--use_aggregate_stereo_vol', action='store_true')
    parser.add_argument('--use_aggregate_mono_vol', action='store_true')
    parser.add_argument('--normal_gain', type=int, default=10)
    parser.add_argument('--lrc_th', type=float, default=1.0)
    parser.add_argument('--iters', type=int, default=32, help='Number of iterations for recurrent networks')

    # MapReduce arguments
    parser.add_argument('--use_mapreduce', type=str, default='none', 
                        choices=['none', 'adaptive', 'tiled', 'memory'],
                        help='MapReduce method: none (default), adaptive, tiled, or memory')
    parser.add_argument('--mapreduce_tile_size', type=int, default=512,
                        help='Tile size for tiled processing (default: 512)')
    parser.add_argument('--mapreduce_tile_overlap', type=int, default=64,
                        help='Tile overlap for tiled processing (default: 64)')
    parser.add_argument('--mapreduce_max_vram_gb', type=float, default=3.5,
                        help='Max VRAM in GB for adaptive/memory methods (default: 3.5)')
    parser.add_argument('--mapreduce_target_scale', type=float, default=0.75,
                        help='Target scale for adaptive resolution (default: 0.75)')

    args = parser.parse_args()

    if args.outdir is not None:
        os.makedirs(args.outdir, exist_ok=True)
        
    dtype = torch.float16 if args.half else torch.float32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
    stereonet = StereoAnywhere(args)

    stereonet = nn.DataParallel(stereonet)
    pretrain_dict = torch.load(args.loadstereomodel, map_location='cpu')
    pretrain_dict = pretrain_dict['state_dict'] if 'state_dict' in pretrain_dict else pretrain_dict
    stereonet.load_state_dict(pretrain_dict, strict=True)  
    stereonet = stereonet.module
    stereonet = stereonet.eval().to(device).to(dtype)

    # Load monocular model
    if args.monomodel == 'DAv2':
        mono_model = get_depth_anything_v2(args.loadmonomodel).eval().to(device).to(dtype)
    else:
        mono_model = None

    # Apply MapReduce wrapper if requested
    if args.use_mapreduce != 'none' and MAPREDUCE_AVAILABLE:
        print(f"\n🔧 Applying MapReduce wrapper: {args.use_mapreduce}")
        
        if args.use_mapreduce == 'adaptive':
            stereonet = AdaptiveResolutionWrapper(
                stereo_model=stereonet,
                mono_model=mono_model,
                max_vram_gb=args.mapreduce_max_vram_gb,
                target_scale=args.mapreduce_target_scale,
                device=device,
                iters=args.iters,
                mixed_precision=args.mixed_precision,
                mono_height=args.mono_height,
                mono_width=args.mono_width
            )
            print(f"   ✅ Adaptive Resolution enabled")
            print(f"      Max VRAM: {args.mapreduce_max_vram_gb} GB")
            print(f"      Target scale: {args.mapreduce_target_scale}")
            
        elif args.use_mapreduce == 'tiled':
            stereonet = TiledStereoWrapper(
                stereo_model=stereonet,
                mono_model=mono_model,
                tile_size=args.mapreduce_tile_size,
                overlap=args.mapreduce_tile_overlap,
                device=device,
                iters=args.iters,
                mixed_precision=args.mixed_precision,
                mono_height=args.mono_height,
                mono_width=args.mono_width
            )
            print(f"   ✅ Tiled Processing enabled")
            print(f"      Tile size: {args.mapreduce_tile_size}x{args.mapreduce_tile_size}")
            print(f"      Overlap: {args.mapreduce_tile_overlap} pixels")
            
        elif args.use_mapreduce == 'memory':
            stereonet = MemoryEfficientWrapper(
                stereo_model=stereonet,
                mono_model=mono_model,
                aggressive_cleanup=True,
                device=device,
                iters=args.iters,
                mono_height=args.mono_height,
                mono_width=args.mono_width
            )
            print(f"   ✅ Memory-Efficient Processing enabled")
            print(f"      Max VRAM: {args.mapreduce_max_vram_gb} GB")
        
        print()
    elif args.use_mapreduce != 'none' and not MAPREDUCE_AVAILABLE:
        print(f"\n⚠️  MapReduce requested but not available. Running without MapReduce.\n")

    if args.display_qualitatives:
        cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)

    for _left, _right in tqdm.tqdm(zip(args.left, args.right), desc="Processing stereo images", total=len(args.left)):
        
        # Load images
        left_image = cv2.imread(_left)
        right_image = cv2.imread(_right)

        if left_image is None:
            print(f"❌ Error: Could not read left image: {_left}")
            continue
        if right_image is None:
            print(f"❌ Error: Could not read right image: {_right}")
            continue

        # check if images are grayscale and convert to BGR
        if len(left_image.shape) == 2:
            left_image = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
        if len(right_image.shape) == 2:
            right_image = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)

        original_shape = left_image.shape

        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)

        # Generate monocular depth
        if mono_model is not None:
            with torch.no_grad():
                with autocast('cuda', enabled=args.mixed_precision):
                    # Prepare images for mono model
                    left_rgb = torch.from_numpy(left_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    right_rgb = torch.from_numpy(right_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    left_rgb = left_rgb.to(device).to(dtype)
                    right_rgb = right_rgb.to(device).to(dtype)
                    
                    # Resize to mono model input size
                    left_mono_input = F.interpolate(left_rgb, size=(args.mono_height, args.mono_width), 
                                                   mode='bilinear', align_corners=False)
                    right_mono_input = F.interpolate(right_rgb, size=(args.mono_height, args.mono_width),
                                                    mode='bilinear', align_corners=False)
                    
                    # Infer mono depth
                    mono_left = mono_model(left_mono_input)
                    mono_right = mono_model(right_mono_input)
                    
                    # Add channel dimension back (mono_model.forward() squeezes it)
                    if mono_left.dim() == 3:  # [B, H, W]
                        mono_left = mono_left.unsqueeze(1)  # [B, 1, H, W]
                    if mono_right.dim() == 3:
                        mono_right = mono_right.unsqueeze(1)
                    
                    # Normalize
                    mono_left = (mono_left - mono_left.min()) / (mono_left.max() - mono_left.min())
                    mono_right = (mono_right - mono_right.min()) / (mono_right.max() - mono_right.min())
                    
                    # Resize to input image size (before iscale)
                    mono_left = F.interpolate(mono_left, size=(original_shape[0], original_shape[1]),
                                            mode='bilinear', align_corners=False)
                    mono_right = F.interpolate(mono_right, size=(original_shape[0], original_shape[1]),
                                             mode='bilinear', align_corners=False)
                    
                    # Save mono visualization if requested
                    if args.save_qualitatives:
                        mono_left_viz = mono_left.squeeze().cpu().numpy()
                        mono_left_viz = (mono_left_viz * 255).astype(np.uint8)
                        mono_left_viz = cv2.applyColorMap(mono_left_viz, cv2.COLORMAP_JET)
                        _output = os.path.join(args.outdir, os.path.splitext(os.path.basename(_left))[0] + '_mono_left.png')
                        cv2.imwrite(_output, mono_left_viz)
        else:
            mono_left = None
            mono_right = None

        # Apply iscale
        if args.iscale != 1.0:
            new_w = round(original_shape[1] / args.iscale)
            new_h = round(original_shape[0] / args.iscale)
            left_image = cv2.resize(left_image, (new_w, new_h))
            right_image = cv2.resize(right_image, (new_w, new_h))
            if mono_left is not None:
                mono_left = F.interpolate(mono_left, size=(new_h, new_w), mode='bilinear', align_corners=False)
                mono_right = F.interpolate(mono_right, size=(new_h, new_w), mode='bilinear', align_corners=False)

        # Prepare inputs
        left_image = (torch.from_numpy(left_image).permute(2, 0, 1).unsqueeze(0) / 255.0).to(device).to(dtype)
        right_image = (torch.from_numpy(right_image).permute(2, 0, 1).unsqueeze(0) / 255.0).to(device).to(dtype)

        # Inference
        with torch.no_grad():
            with autocast('cuda', enabled=args.mixed_precision):
                if args.use_mapreduce != 'none' and MAPREDUCE_AVAILABLE:
                    # MapReduce wrappers handle mono input and iters internally
                    pred_disp = stereonet(left_image, right_image, mono_left, mono_right)
                else:
                    # Direct inference (original model)
                    pred_disp = stereonet(left_image, right_image, mono_left, mono_right, iters=args.iters)
        
        pred_disp = pred_disp.detach().squeeze().float().cpu().numpy()
        
        # Resize back to original if iscale was used
        if args.iscale != 1.0:
            pred_disp = cv2.resize(pred_disp, (original_shape[1], original_shape[0]))

        # Save the output
        _output = f"{os.path.splitext(_left)[0]}.npy" if args.outdir is None else os.path.join(args.outdir, os.path.splitext(os.path.basename(_left))[0] + '.npy')
        np.save(_output, pred_disp)
        print(f"✅ Saved: {_output}")
        
        if args.save_qualitatives or args.display_qualitatives:
            pred_disp_viz = (pred_disp - pred_disp.min()) / (pred_disp.max() - pred_disp.min())
            pred_disp_viz = (pred_disp_viz * 255).astype(np.uint8)
            pred_disp_viz = cv2.applyColorMap(pred_disp_viz, cv2.COLORMAP_JET)

            if args.save_qualitatives:
                _output = f"{os.path.splitext(_left)[0]}_disp_jet.png" if args.outdir is None else os.path.join(args.outdir, os.path.splitext(os.path.basename(_left))[0] + '_disp_jet.png')
                cv2.imwrite(_output, pred_disp_viz)
                print(f"✅ Saved visualization: {_output}")

            if args.display_qualitatives:
                cv2.imshow("Disparity", pred_disp_viz)
                cv2.waitKey(1)
                
    if args.display_qualitatives:
        cv2.destroyAllWindows()

    print("\n✅ Processing complete!")

        
if __name__ == "__main__":
    main()
