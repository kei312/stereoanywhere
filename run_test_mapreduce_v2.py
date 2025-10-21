import os
import subprocess
from pathlib import Path

# ==============================================================================
# CẤU HÌNH CƠ BẢN
# ==============================================================================
PROJECT_ROOT = Path('/mnt/c/BTL PTDDL/stereoanywhere')
DATAPATH = PROJECT_ROOT / 'datasets/mb2014/trainingH'
DATASET_NAME = 'middlebury'
STEREO_MODEL_PATH = PROJECT_ROOT / 'pretrained/sceneflow.tar'
MONO_MODEL_PATH = PROJECT_ROOT / 'pretrained/depth_anything_v2_vitl.pth'
ISCALE = 1  # MapReduce cho phép xử lý full resolution
OSCALE = 1

# ==============================================================================
# CẤU HÌNH TILE - CHỌN MỘT TRONG CÁC CÁCH SAU
# ==============================================================================

# CÁCH 1: Sử dụng Preset (Khuyến nghị)
# Các preset có sẵn: 'default', 'middlebury', 'kitti', 'sceneflow', 'booster', 
#                    'monotrap', 'small_image', 'large_image', 'low_memory', 'high_memory'
# Dùng 'auto' để tự động chọn dựa trên DATASET_NAME
# Dùng 'list' để xem danh sách tất cả preset
USE_PRESET = True
TILE_PRESET = 'middlebury'  # Hoặc 'auto' để tự chọn, hoặc 'list' để xem danh sách

# CÁCH 2: Tùy chỉnh thủ công (nếu USE_PRESET = False)
# Tile hình chữ nhật
USE_RECTANGULAR_TILES = True  # True = hình chữ nhật, False = hình vuông
TILE_SIZE = 768      # Chỉ dùng nếu USE_RECTANGULAR_TILES = False
TILE_WIDTH = 768     # Chiều rộng tile (chỉ dùng khi USE_RECTANGULAR_TILES = True)
TILE_HEIGHT = 512    # Chiều cao tile (chỉ dùng khi USE_RECTANGULAR_TILES = True)
OVERLAP = 96         # Độ chồng lấp

# ==============================================================================
# CẤU HÌNH GLOBAL GUIDANCE (Context-Aware MapReduce)
# ==============================================================================

# Bật/tắt global guidance tự động
# DISABLED BY DEFAULT: Guidance causes 6x slowdown and may worsen results
# Only enable if you explicitly need it for specific datasets
USE_GLOBAL_GUIDANCE = False  # True = bật guidance, False = tắt (MapReduce thuần túy)
GUIDANCE_SCALE = 2.0         # Tỷ lệ scale cho low-res pass (2.0 = downscale x2)
GUIDANCE_WEIGHT = 0.3        # Trọng số ảnh hưởng của guidance (0-1)

# LƯU Ý: Nếu sử dụng preset (USE_PRESET = True), các giá trị TILE_WIDTH, 
#        TILE_HEIGHT, OVERLAP ở trên sẽ bị ghi đè bởi giá trị từ preset.
#        Bạn có thể chỉnh preset trực tiếp trong file: 
#        mapreduce_v2/tile_presets.py

# ==============================================================================
# KHÔNG SỬA DƯỚI ĐÂY
# ==============================================================================
if not PROJECT_ROOT.is_dir():
    raise SystemExit(f"PROJECT_ROOT không tồn tại: {PROJECT_ROOT}")

os.chdir(PROJECT_ROOT)

command = [
    'python', 'test_mapreduce_v2.py',
    '--datapath', str(DATAPATH),
    '--dataset', DATASET_NAME,
    '--loadstereomodel', str(STEREO_MODEL_PATH),
    '--loadmonomodel', str(MONO_MODEL_PATH),
    '--iscale', str(ISCALE),
    '--oscale', str(OSCALE),
    '--mixed_precision',
    '--use_truncate_vol',
    '--use_aggregate_mono_vol',
    '--mirror_conf_th', '0.95',
    '--mirror_attenuation', '0.85'
]

# Thêm cấu hình tile dựa trên lựa chọn
if USE_PRESET:
    command.extend(['--tile_preset', TILE_PRESET])
elif USE_RECTANGULAR_TILES:
    command.extend([
        '--tile_width', str(TILE_WIDTH),
        '--tile_height', str(TILE_HEIGHT),
        '--overlap', str(OVERLAP)
    ])
else:
    command.extend([
        '--tile_size', str(TILE_SIZE),
        '--overlap', str(OVERLAP)
    ])

# Thêm cấu hình global guidance nếu bật
if USE_GLOBAL_GUIDANCE:
    command.extend([
        '--use_global_guidance',
        '--guidance_scale', str(GUIDANCE_SCALE),
        '--guidance_weight', str(GUIDANCE_WEIGHT)
    ])

print('Running MapReduce evaluation:')
print(' '.join(command))

try:
    subprocess.run(command, check=True)
except subprocess.CalledProcessError as exc:
    print(f"MapReduce test thất bại: {exc}")