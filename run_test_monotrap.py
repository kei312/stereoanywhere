import subprocess
import os

# ==============================================================================
# FILE NÀY CHỈ DÙNG ĐỂ CHẠY DATASET MONOTRAP
# ==============================================================================

PROJECT_ROOT = r'C:\BTL PTDDL\stereoanywhere'
DATAPATH = r'C:\BTL PTDDL\stereoanywhere\datasets\MonoTrap\validation'
DATASET_NAME = 'MonoTrap'
STEREO_MODEL_PATH = r'C:\BTL PTDDL\stereoanywhere\pretrained\sceneflow.tar'
MONO_MODEL_PATH = r'C:\BTL PTDDL\stereoanywhere\pretrained\depth_anything_v2_vitl.pth'
ISCALE = 1
OSCALE = 1

# ==============================================================================
# KHÔNG CẦN CHỈNH SỬA PHẦN DƯỚI ĐÂY
# ==============================================================================

os.chdir(PROJECT_ROOT)

# Xây dựng lệnh test - LƯU Ý: gọi file test_monotrap.py
command = [
    'python', 'test_monotrap.py', # <--- ĐIỂM KHÁC BIỆT CỐT LÕI
    '--datapath', DATAPATH,
    '--dataset', DATASET_NAME,
    '--stereomodel', 'stereoanywhere',
    '--loadstereomodel', STEREO_MODEL_PATH,
    '--monomodel', 'DAv2',
    '--loadmonomodel', MONO_MODEL_PATH,
    '--iscale', str(ISCALE),
    '--oscale', str(OSCALE),
    '--normalize',
    '--iters', '32',
    '--vol_n_masks', '8',
    '--n_additional_hourglass', '0',
    '--use_aggregate_mono_vol',
    '--vol_downsample', '0',
    '--mirror_conf_th', '0.98',
    '--use_truncate_vol',
    '--mirror_attenuation', '0.9'
]

print(f"Đang chạy lệnh cho dataset MonoTrap:\n{' '.join(command)}\n")

try:
    subprocess.run(command, check=True)
    print("\nScript test cho MonoTrap đã chạy thành công!")
except subprocess.CalledProcessError as e:
    print(f"\nLỗi khi chạy script test: {e}")