#!/bin/bash
# ===============================================================================
# Script chạy demo StereoAnywhere
# ===============================================================================
# Cách dùng: ./run_demo.sh <left_image> <right_image>
# 
# Ví dụ:
#   ./run_demo.sh datasets/mb2014/trainingH/Adirondack/im0.png datasets/mb2014/trainingH/Adirondack/im1.png
#
# Hoặc để dùng ảnh mặc định, chỉnh DEFAULT_LEFT và DEFAULT_RIGHT ở dưới và chạy:
#   ./run_demo.sh
# ===============================================================================

# =========================
# PHẦN 1: ẢNH ĐẦU VÀO (Tùy chọn)
# =========================
# Nếu không truyền tham số khi chạy script, sẽ dùng ảnh mặc định này:
DEFAULT_LEFT="datasets/layeredflow_val/20/0_0.png"
DEFAULT_RIGHT="datasets/layeredflow_val/20/0_1.png"
#"C:\BTL PTDDL\stereoanywhere\datasets\layeredflow_val\18\0_0.png"
# =========================
# PHẦN 2: PRETRAINED MODELS
# =========================
STEREO_MODEL="pretrained/sceneflow.tar"
MONO_MODEL="pretrained/depth_anything_v2_vitl.pth"

# =========================
# PHẦN 3: THÔNG SỐ CHẠY
# =========================
# Thư mục lưu kết quả
OUTPUT_DIR="demo_output"

# Thông số xử lý ảnh
ISCALE=16              # Tỷ lệ resize ảnh (1=không resize, 2=giảm 1/2, 4=giảm 1/4)
                      # Tăng giá trị này nếu gặp lỗi Out of Memory
                      # Khuyến nghị: 1 (GPU >8GB), 2 (GPU 6-8GB), 4 (GPU 4GB)

ITERS=64              # Số vòng lặp RAFT (8-32)
                      # Nhiều hơn = chính xác hơn nhưng chậm hơn
                      # Khuyến nghị: 32 (chất lượng cao), 16 (cân bằng), 8 (nhanh)

VOL_DOWNSAMPLE=0     # Giảm kích thước cost volume (0-2)
                      # Tăng giá trị này nếu gặp lỗi Out of Memory
                      # Khuyến nghị: 0 (GPU >8GB), 1 (GPU 6GB), 2 (GPU 4GB)

VOL_N_MASKS=8         # Số lượng masks cho cost volume (4-8)
VOLUME_CHANNELS=8     # Số channels của volume (8)
MIRROR_CONF_TH=0.98   # Ngưỡng confidence cho mirror detection (0.95-0.99)
MIRROR_ATTENUATION=0.9 # Hệ số giảm cho mirror regions (0.8-0.95)

# =========================
# KHÔNG CẦN SỬA PHÍA DƯỚI
# =========================

# Xử lý tham số đầu vào
if [ "$#" -eq 0 ]; then
    # Không có tham số -> dùng ảnh mặc định
    LEFT_IMAGE="$DEFAULT_LEFT"
    RIGHT_IMAGE="$DEFAULT_RIGHT"
    echo "ℹ️  Không có tham số -> Dùng ảnh mặc định"
elif [ "$#" -eq 2 ]; then
    # Có 2 tham số -> dùng ảnh do user cung cấp
    LEFT_IMAGE="$1"
    RIGHT_IMAGE="$2"
else
    # Sai số tham số
    echo "❌ Lỗi: Cần 0 hoặc 2 tham số"
    echo ""
    echo "Cách dùng:"
    echo "  1. Chạy với ảnh mặc định:"
    echo "     ./run_demo.sh"
    echo ""
    echo "  2. Chạy với ảnh tùy chọn:"
    echo "     ./run_demo.sh <left_image> <right_image>"
    echo ""
    echo "Ví dụ:"
    echo "  ./run_demo.sh datasets/mb2014/trainingH/Motorcycle/im0.png datasets/mb2014/trainingH/Motorcycle/im1.png"
    exit 1
fi

# Kiểm tra file tồn tại
if [ ! -f "$LEFT_IMAGE" ]; then
    echo "Lỗi: Không tìm thấy ảnh trái: $LEFT_IMAGE"
    exit 1
fi

if [ ! -f "$RIGHT_IMAGE" ]; then
    echo "Lỗi: Không tìm thấy ảnh phải: $RIGHT_IMAGE"
    exit 1
fi

# Tạo tên thư mục con từ đường dẫn ảnh
# Ví dụ: datasets/mb2014/trainingH/Adirondack/im0.png
#     -> mb2014_trainingH_Adirondack
IMG_PATH=$(dirname "$LEFT_IMAGE")
# Lấy 3 cấp cuối của đường dẫn và nối bằng dấu _
SCENE_NAME=$(echo "$IMG_PATH" | awk -F'/' '{
    n=NF
    if (n >= 3) {
        printf "%s_%s_%s", $(n-2), $(n-1), $n
    } else if (n == 2) {
        printf "%s_%s", $(n-1), $n
    } else {
        printf "%s", $n
    }
}')

# Nếu không parse được tên, dùng tên thư mục cuối
if [ -z "$SCENE_NAME" ]; then
    SCENE_NAME=$(basename "$IMG_PATH")
fi

# Tạo thư mục output cho scene này
OUTPUT_DIR_FULL="${OUTPUT_DIR}/${SCENE_NAME}"
mkdir -p "$OUTPUT_DIR_FULL"

echo "=========================================="
echo "StereoAnywhere Demo"
echo "=========================================="
echo "📁 Ảnh trái:  $LEFT_IMAGE"
echo "📁 Ảnh phải:  $RIGHT_IMAGE"
echo "🤖 Stereo model: $STEREO_MODEL"
echo "🤖 Mono model:   $MONO_MODEL"
echo "💾 Output:    $OUTPUT_DIR_FULL"
echo "📂 Scene:     $SCENE_NAME"
echo ""
echo "⚙️  Thông số:"
echo "   - iscale: $ISCALE (resize ảnh đầu vào)"
echo "   - iters: $ITERS (số vòng lặp)"
echo "   - vol_downsample: $VOL_DOWNSAMPLE (giảm cost volume)"
echo "=========================================="
echo ""

# Copy ảnh gốc vào output folder
echo "📋 Copying input images..."
cp "$LEFT_IMAGE" "$OUTPUT_DIR_FULL/$(basename $LEFT_IMAGE)"
cp "$RIGHT_IMAGE" "$OUTPUT_DIR_FULL/$(basename $RIGHT_IMAGE)"

# Chạy demo
cd demo
python fast_demo.py \
    --left "../$LEFT_IMAGE" \
    --right "../$RIGHT_IMAGE" \
    --outdir "../$OUTPUT_DIR_FULL" \
    --stereomodel stereoanywhere \
    --loadstereomodel "../$STEREO_MODEL" \
    --monomodel DAv2 \
    --loadmonomodel "../$MONO_MODEL" \
    --iscale $ISCALE \
    --iters $ITERS \
    --vol_n_masks $VOL_N_MASKS \
    --volume_channels $VOLUME_CHANNELS \
    --n_additional_hourglass 0 \
    --use_aggregate_mono_vol \
    --vol_downsample $VOL_DOWNSAMPLE \
    --mirror_conf_th $MIRROR_CONF_TH \
    --use_truncate_vol \
    --mirror_attenuation $MIRROR_ATTENUATION \
    --save_qualitatives \
    --mixed_precision

cd ..

echo ""
echo "=========================================="
echo "✅ Hoàn thành! Kết quả được lưu tại: $OUTPUT_DIR_FULL"
echo "=========================================="
echo ""
echo "📊 Kết quả bao gồm:"
echo "   - $(basename $LEFT_IMAGE) - Ảnh trái gốc"
echo "   - $(basename $RIGHT_IMAGE) - Ảnh phải gốc"
echo "   - *.npy - Dữ liệu disparity thô"
echo "   - *_disp_jet.png - Ảnh disparity với màu sắc"
echo "   - *_mono_left.png - Monocular depth (từ Depth Anything V2)"
echo ""
echo "📂 Xem kết quả:"
echo "   ls -lh $OUTPUT_DIR_FULL/"
echo "   eog $OUTPUT_DIR_FULL/*_disp_jet.png"
echo ""
echo "💡 Tips:"
echo "   - Nếu gặp lỗi Out of Memory, tăng ISCALE hoặc VOL_DOWNSAMPLE"
echo "   - Để tăng chất lượng, tăng ITERS (nhưng sẽ chậm hơn)"
echo "=========================================="
