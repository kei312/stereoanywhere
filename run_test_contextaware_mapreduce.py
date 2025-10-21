#!/usr/bin/env python3
"""
Two-pass MapReduce processing with global guidance
Pass 1: Low-res để lấy global context
Pass 2: High-res MapReduce sử dụng guidance từ Pass 1
"""
import os
import numpy as np
import torch
import cv2
from pathlib import Path
import subprocess
import argparse

# ==============================================================================
# CẤU HÌNH CƠ BẢN
# ==============================================================================
PROJECT_ROOT = Path('/mnt/c/BTL PTDDL/stereoanywhere')
DATAPATH = PROJECT_ROOT / 'datasets/mb2014/trainingH'
DATASET_NAME = 'middlebury'
STEREO_MODEL_PATH = PROJECT_ROOT / 'pretrained/sceneflow.tar'
MONO_MODEL_PATH = PROJECT_ROOT / 'pretrained/depth_anything_v2_vitl.pth'

# Thông số cấu hình two-pass
LOW_RES_SCALE = 2.0    # Scale cho global context (pass 1)
HIGH_RES_SCALE = 1.0   # Scale cho full resolution MapReduce (pass 2)
TILE_PRESET = 'middlebury'  # Hoặc thủ công với TILE_WIDTH, TILE_HEIGHT
GUIDANCE_WEIGHT = 0.3  # Trọng số của global guidance (0-1)

OUTPUT_DIR = PROJECT_ROOT / 'output_contextaware'
TEMP_DIR = OUTPUT_DIR / 'temp'

# ==============================================================================
# KHÔNG SỬA DƯỚI ĐÂY
# ==============================================================================

def setup_directories():
    """Tạo thư mục output"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR / 'lowres', exist_ok=True)
    os.makedirs(TEMP_DIR / 'guidance', exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Temp directory: {TEMP_DIR}")


def run_lowres_pass():
    """Pass 1: Chạy low-res để lấy global context"""
    print("\n" + "="*80)
    print("[PASS 1/2] Running low-resolution pass for global context...")
    print("="*80)
    
    os.chdir(PROJECT_ROOT)
    
    command = [
        'python', 'test.py',
        '--datapath', str(DATAPATH),
        '--dataset', DATASET_NAME,
        '--stereomodel', 'stereoanywhere',
        '--loadstereomodel', str(STEREO_MODEL_PATH),
        '--monomodel', 'DAv2',
        '--loadmonomodel', str(MONO_MODEL_PATH),
        '--iscale', str(LOW_RES_SCALE),
        '--oscale', str(LOW_RES_SCALE),
        '--outdir', str(TEMP_DIR / 'lowres'),
        '--mixed_precision',
        '--use_truncate_vol',
        '--use_aggregate_mono_vol',
        '--iters', '32',
        '--vol_n_masks', '8'
    ]
    
    print("\nCommand:")
    print(' '.join(command))
    print()
    
    try:
        subprocess.run(command, check=True)
        print(f"\n✓ Low-res pass completed. Results in {TEMP_DIR / 'lowres'}")
        return TEMP_DIR / 'lowres'
    except subprocess.CalledProcessError as e:
        print(f"✗ Low-res pass failed: {e}")
        raise


def prepare_guidance(lowres_dir):
    """Chuẩn bị global guidance từ kết quả low-res"""
    print("\n" + "="*80)
    print("[PREPARATION] Creating global guidance maps...")
    print("="*80)
    
    guidance_maps = {}
    
    # Tìm tất cả các file disparity từ low-res output
    raw_dir = lowres_dir / 'raw'
    if not raw_dir.exists():
        print(f"✗ Raw directory not found: {raw_dir}")
        return guidance_maps
    
    npy_files = list(raw_dir.glob('*.npy'))
    print(f"\nFound {len(npy_files)} disparity maps")
    
    for npy_file in npy_files:
        scene_name = npy_file.stem
        
        try:
            # Đọc disparity map từ low-res
            disp_map = np.load(str(npy_file))
            
            # Loại bỏ chiều batch nếu có
            if disp_map.ndim == 4:
                disp_map = disp_map.squeeze(0).squeeze(0)
            elif disp_map.ndim == 3:
                disp_map = disp_map.squeeze(0)
            
            h, w = disp_map.shape
            print(f"  {scene_name}: {w}x{h} -> ", end='')
            
            # Upscale về kích thước gốc
            new_h = int(h * LOW_RES_SCALE)
            new_w = int(w * LOW_RES_SCALE)
            
            disp_map_upscaled = cv2.resize(
                disp_map, 
                (new_w, new_h), 
                interpolation=cv2.INTER_LINEAR
            )
            
            # Điều chỉnh giá trị disparity theo tỷ lệ scale
            disp_map_upscaled *= LOW_RES_SCALE
            
            # Lưu guidance map
            guidance_file = TEMP_DIR / 'guidance' / f'{scene_name}_guidance.npy'
            np.save(guidance_file, disp_map_upscaled)
            guidance_maps[scene_name] = guidance_file
            
            print(f"{new_w}x{new_h} ✓")
            
        except Exception as e:
            print(f"  {scene_name}: Error - {e}")
            continue
    
    print(f"\n✓ Created {len(guidance_maps)} guidance maps")
    return guidance_maps


def run_highres_mapreduce_with_guidance(guidance_maps):
    """Pass 2: Chạy high-res MapReduce với global guidance"""
    print("\n" + "="*80)
    print("[PASS 2/2] Running high-resolution MapReduce with global guidance...")
    print("="*80)
    
    os.chdir(PROJECT_ROOT)
    
    # Tạo temporary file chứa paths của guidance maps
    guidance_dir = TEMP_DIR / 'guidance'
    
    command = [
        'python', 'test_mapreduce_v2.py',
        '--datapath', str(DATAPATH),
        '--dataset', DATASET_NAME,
        '--loadstereomodel', str(STEREO_MODEL_PATH),
        '--loadmonomodel', str(MONO_MODEL_PATH),
        '--iscale', str(HIGH_RES_SCALE),
        '--oscale', str(HIGH_RES_SCALE),
        '--tile_preset', TILE_PRESET,
        '--outdir', str(OUTPUT_DIR),
        '--mixed_precision',
        '--use_truncate_vol',
        '--use_aggregate_mono_vol',
        '--mirror_conf_th', '0.95',
        '--mirror_attenuation', '0.85',
        '--iters', '32',
        '--vol_n_masks', '8',
        '--guidance_dir', str(guidance_dir),
        '--guidance_weight', str(GUIDANCE_WEIGHT)
    ]
    
    print("\nCommand:")
    print(' '.join(command))
    print()
    
    try:
        subprocess.run(command, check=True)
        print(f"\n✓ High-res MapReduce completed. Results in {OUTPUT_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"✗ High-res MapReduce failed: {e}")
        raise


def main():
    """Thực hiện quy trình hai bước với global guidance"""
    print("\n" + "="*80)
    print("CONTEXT-AWARE MAPREDUCE PROCESSING")
    print("="*80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Data path: {DATAPATH}")
    print(f"Low-res scale: {LOW_RES_SCALE}x")
    print(f"High-res scale: {HIGH_RES_SCALE}x")
    print(f"Tile preset: {TILE_PRESET}")
    print(f"Guidance weight: {GUIDANCE_WEIGHT}")
    print("="*80)
    
    # Setup directories
    setup_directories()
    
    # Pass 1: Low-res để lấy global context
    lowres_dir = run_lowres_pass()
    
    # Preparation: Tạo guidance maps
    guidance_maps = prepare_guidance(lowres_dir)
    
    if not guidance_maps:
        print("\n✗ No guidance maps created. Aborting Pass 2.")
        return
    
    # Pass 2: High-res MapReduce với guidance
    run_highres_mapreduce_with_guidance(guidance_maps)
    
    print("\n" + "="*80)
    print("✓ COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*80)


if __name__ == "__main__":
    main()
