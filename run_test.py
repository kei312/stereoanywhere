import subprocess
import os

# ==============================================================================
# THÔNG TIN BẠN CẦN ĐIỀN VÀO ĐÂY
# ==============================================================================

# 1. Đường dẫn đến thư mục gốc của dự án "Stereo Anywhere"
PROJECT_ROOT = r'/mnt/c/BTL PTDDL/stereoanywhere'    #linux wsl
# r'C:\BTL PTDDL\stereoanywhere'                    

# 2. Đường dẫn đến thư mục chứa bộ dữ liệu BẠN MUỐN TEST
#    Ví dụ: r'C:\BTL PTDDL\stereoanywhere\datasets\Middlebury'          booster_gt/train
#    Ví dụ: r'C:\BTL PTDDL\stereoanywhere\datasets\KITTI2015\testing'
DATAPATH = r'/mnt/c/BTL PTDDL/stereoanywhere/datasets/mb2014/trainingH' # <--- THAY ĐỔI Ở ĐÂY   #linux wsl

# 3. Tên của bộ dữ liệu (phải khớp với DATAPATH ở trên)
#    Các lựa chọn: 'middlebury (aka middlebury2014)', 'middlebury2021', 'eth3d', 'kitti2012', 'kitti2015', 'booster', 'layeredflow'
DATASET_NAME = 'middlebury' # <--- THAY ĐỔI Ở ĐÂY

# 4. Đường dẫn đến mô hình stereo đã được huấn luyện (.tar)
STEREO_MODEL_PATH = r'/mnt/c/BTL PTDDL/stereoanywhere/pretrained/sceneflow.tar' #linux wsl
# r'C:\BTL PTDDL\stereoanywhere\pretrained\sceneflow.tar'
#r'C:\BTL PTDDL\stereoanywhere\pretrained\checkpoint_mono_100_booster.pth'
#booster có model khác trong drive :(và 1 số cái khác cũng trong file drive)

# 5. Đường dẫn đến mô hình monocular DAv2 đã được huấn luyện (.pth)
MONO_MODEL_PATH = r'/mnt/c/BTL PTDDL/stereoanywhere/pretrained/depth_anything_v2_vitl.pth' #linux wsl
# r'C:\BTL PTDDL\pretrained\depth_anything_v2_vitl.pth'

# 6. Hệ số tỷ lệ đầu vào (ISCALE) - RẤT QUAN TRỌNG
#    - Dùng 4 cho dataset 'booster'
#    - Dùng 8 cho dataset 'layeredflow'
#    - Dùng 1 cho tất cả các dataset còn lại (middlebury, kitti, eth3d)
ISCALE = 8# <--- THAY ĐỔI Ở ĐÂY NẾU CẦN

# 7. Hệ số tỷ lệ đầu ra (OSCALE) - Thường giống ISCALE
OSCALE = 8# <--- THAY ĐỔI Ở ĐÂY NẾU CẦN

# ==============================================================================
# KHÔNG CẦN CHỈNH SỬA PHẦN DƯỚI ĐÂY
# ==============================================================================

# Đảm bảo đường dẫn dự án tồn tại
if not os.path.isdir(PROJECT_ROOT):
    print(f"Lỗi: Đường dẫn PROJECT_ROOT không tồn tại: {PROJECT_ROOT}")
    exit()

# Chuyển đến thư mục gốc của dự án
os.chdir(PROJECT_ROOT)

# Xây dựng lệnh test
command = [
    'python', 'test.py',
    '--datapath', DATAPATH,
    '--dataset', DATASET_NAME,
    '--stereomodel', 'stereoanywhere',
    '--loadstereomodel', STEREO_MODEL_PATH,
    '--monomodel', 'DAv2',
    '--loadmonomodel', MONO_MODEL_PATH,

    # THÊM DÒNG NÀY VÀO:
    '--preload_mono',
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

print(f"Đang chuẩn bị chạy lệnh cho dataset '{DATASET_NAME}':\n{' '.join(command)}\n")

try:
    # Chạy lệnh
    subprocess.run(command, check=True)
    print(f"\nScript test cho dataset '{DATASET_NAME}' đã chạy thành công!")
except subprocess.CalledProcessError as e:
    print(f"\nLỗi khi chạy script test: {e}")
    if e.stderr:
        print(f"Output lỗi:\n{e.stderr.decode('utf-8')}")
except FileNotFoundError:
    print("\nLỗi: Không tìm thấy 'python' hoặc 'test.py'.")
    print("Đảm bảo Python đã được cài đặt và bạn đang ở đúng thư mục dự án.")