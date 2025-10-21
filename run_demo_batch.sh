#!/bin/bash
# ===============================================================================
# Script chạy batch demo cho nhiều scene
# ===============================================================================
# CÁCH DÙNG NHANH: 
#   1. Uncomment MỘT preset bên dưới (bỏ dấu #)
#   2. Chạy: ./run_demo_batch.sh
# ===============================================================================

# =========================
# 🎯 CHỌN PRESET DATASET (Uncomment 1 dòng để dùng)
# =========================

# ⭐ BOOSTER - 5 scene đầu tiên (mặc định)
PRESET="booster_5"

# BOOSTER - Tất cả scene (auto scan folder)
# PRESET="booster_all"

# MIDDLEBURY 2014 - Test set (15 scenes)
# PRESET="mb2014_test"

# MIDDLEBURY 2014 - Training set (15 scenes)
# PRESET="mb2014_train"

# MIDDLEBURY 2021 - Tất cả (20 scenes)
# PRESET="mb2021_all"

# Tùy chỉnh thủ công (xem cuối file)
# PRESET="custom"

# =========================
# KHÔNG CẦN SỬA PHÍA DƯỚI (trừ khi dùng "custom")
# =========================

# =========================
# KHÔNG CẦN SỬA PHÍA DƯỚI (trừ khi dùng "custom")
# =========================

# Load preset configuration
case "$PRESET" in
    "booster_5")
        echo "📦 Preset: Booster - 5 scene đầu tiên"
        DATASET_TYPE="booster"
        AUTO_SCAN_DIR=""
        SCENES=(
            "datasets/booster_gt/train/balanced/Bathroom"
            "datasets/booster_gt/train/balanced/Bedroom"
            "datasets/booster_gt/train/balanced/Bottle"
            "datasets/booster_gt/train/balanced/Bottle1"
            "datasets/booster_gt/train/balanced/BottledWater"
        )
        ;;
    "booster_all")
        echo "📦 Preset: Booster - Tất cả scene (auto scan)"
        DATASET_TYPE="booster"
        AUTO_SCAN_DIR="datasets/booster_gt/train/balanced"
        SCENES=()
        ;;
    "mb2014_test")
        echo "📦 Preset: Middlebury 2014 Test (15 scenes)"
        DATASET_TYPE="middlebury"
        AUTO_SCAN_DIR="datasets/mb2014/testH"
        SCENES=()
        ;;
    "mb2014_train")
        echo "📦 Preset: Middlebury 2014 Training (15 scenes)"
        DATASET_TYPE="middlebury"
        AUTO_SCAN_DIR="datasets/mb2014/trainingH"
        SCENES=()
        ;;
    "mb2021_all")
        echo "📦 Preset: Middlebury 2021 (20 scenes)"
        DATASET_TYPE="middlebury"
        AUTO_SCAN_DIR="datasets/mb2021"
        SCENES=()
        ;;
    "custom")
        echo "📦 Preset: Custom (xem cuối file để cấu hình)"
        # Default values - sẽ được ghi đè bởi CUSTOM_CONFIG nếu được define
        DATASET_TYPE="middlebury"
        AUTO_SCAN_DIR=""
        SCENES=()
        
        # Load custom config nếu có
        if declare -f CUSTOM_CONFIG > /dev/null; then
            CUSTOM_CONFIG
        else
            echo "⚠️  Chưa cấu hình CUSTOM_CONFIG"
            echo "    Xem cuối file run_demo_batch.sh để cấu hình"
        fi
        ;;
    *)
        echo "❌ PRESET không hợp lệ: $PRESET"
        echo "Các preset hợp lệ: booster_5, booster_all, mb2014_test, mb2014_train, mb2021_all, custom"
        exit 1
        ;;
esac

# =========================
# PHẦN 2: PRETRAINED MODELS
# =========================
STEREO_MODEL="pretrained/sceneflow.tar"
MONO_MODEL="pretrained/depth_anything_v2_vitl.pth"

# =========================
# PHẦN 3: THÔNG SỐ CHẠY
# =========================
OUTPUT_DIR="demo_output"
ISCALE=8
ITERS=32
VOL_DOWNSAMPLE=2
VOL_N_MASKS=8   #32   #
VOLUME_CHANNELS=8
MIRROR_CONF_TH=0.98
MIRROR_ATTENUATION=0.9

# =========================
# KHÔNG CẦN SỬA PHÍA DƯỚI
# =========================

# Nếu AUTO_SCAN_DIR được set, tự động tìm tất cả scene
if [ -n "$AUTO_SCAN_DIR" ] && [ -d "$AUTO_SCAN_DIR" ]; then
    echo "🔍 Đang scan folder: $AUTO_SCAN_DIR"
    echo ""
    
    SCENES=()
    for dir in "$AUTO_SCAN_DIR"/*; do
        if [ -d "$dir" ]; then
            LEFT_FILE="$dir/$LEFT_NAME"
            RIGHT_FILE="$dir/$RIGHT_NAME"
            
            if [ -f "$LEFT_FILE" ] && [ -f "$RIGHT_FILE" ]; then
                SCENES+=("$dir")
                echo "  ✓ Found: $(basename $dir)"
            else
                echo "  ⊗ Skip: $(basename $dir) (missing images)"
            fi
        fi
    done
    
    echo ""
    
    if [ ${#SCENES[@]} -eq 0 ]; then
        echo "❌ Không tìm thấy scene nào trong $AUTO_SCAN_DIR"
        echo ""
        echo "💡 Hướng dẫn:"
        echo "   1. Kiểm tra đường dẫn: ls -la $AUTO_SCAN_DIR"
        echo "   2. Kiểm tra tên file ảnh: LEFT_NAME=$LEFT_NAME, RIGHT_NAME=$RIGHT_NAME"
        echo "   3. Hoặc dùng SCENES array thủ công (xem script)"
        exit 1
    fi
fi

# Kiểm tra SCENES array có scene không
if [ ${#SCENES[@]} -eq 0 ]; then
    echo "❌ Lỗi: Không có scene nào để chạy"
    echo ""
    echo "💡 Hướng dẫn:"
    echo "   1. Dùng AUTO_SCAN_DIR (khuyến nghị):"
    echo "      AUTO_SCAN_DIR=\"datasets/mb2014/testH\""
    echo ""
    echo "   2. Hoặc dùng SCENES array thủ công:"
    echo "      SCENES=("
    echo "          \"datasets/mb2014/testH/Adirondack\""
    echo "          \"datasets/mb2014/testH/Australia\""
    echo "      )"
    exit 1
fi

echo "=========================================="
echo "StereoAnywhere Batch Demo"
echo "=========================================="
if [ -n "$AUTO_SCAN_DIR" ]; then
    echo "📂 Auto scan: $AUTO_SCAN_DIR"
fi
echo "📊 Số scene: ${#SCENES[@]}"
echo "⚙️  Thông số: iscale=$ISCALE, iters=$ITERS, vol_downsample=$VOL_DOWNSAMPLE"
echo "=========================================="
echo ""

# Đếm số scene thành công và thất bại
SUCCESS=0
FAILED=0
FAILED_SCENES=()

# Lặp qua từng scene
for i in "${!SCENES[@]}"; do
    SCENE="${SCENES[$i]}"
    
    # Xác định đường dẫn ảnh dựa trên dataset type
    if [ "$DATASET_TYPE" = "booster" ]; then
        # Booster: camera_00/im0.png (left) và camera_02/im0.png (right)
        LEFT_IMAGE="${SCENE}/camera_00/im0.png"
        RIGHT_IMAGE="${SCENE}/camera_02/im0.png"
    else
        # Middlebury: im0.png và im1.png trực tiếp
        LEFT_IMAGE="${SCENE}/${LEFT_NAME}"
        RIGHT_IMAGE="${SCENE}/${RIGHT_NAME}"
    fi
    
    SCENE_NUM=$((i + 1))
    TOTAL=${#SCENES[@]}
    
    echo "=========================================="
    echo "[$SCENE_NUM/$TOTAL] Processing: $(basename $SCENE)"
    echo "=========================================="
    
    # Kiểm tra file tồn tại
    if [ ! -f "$LEFT_IMAGE" ]; then
        echo "❌ Không tìm thấy: $LEFT_IMAGE"
        FAILED=$((FAILED + 1))
        FAILED_SCENES+=("$(basename $SCENE) - Left image not found")
        continue
    fi
    
    if [ ! -f "$RIGHT_IMAGE" ]; then
        echo "❌ Không tìm thấy: $RIGHT_IMAGE"
        FAILED=$((FAILED + 1))
        FAILED_SCENES+=("$(basename $SCENE) - Right image not found")
        continue
    fi
    
    # Tạo tên thư mục output
    if [ "$DATASET_TYPE" = "booster" ]; then
        # Booster: Lấy từ SCENE path
        # Ví dụ: datasets/booster_gt/train/balanced/Bathroom
        #     -> booster_gt_train_balanced_Bathroom
        SCENE_NAME=$(echo "$SCENE" | awk -F'/' '{
            n=NF
            if (n >= 4) {
                printf "%s_%s_%s_%s", $(n-3), $(n-2), $(n-1), $n
            } else if (n >= 3) {
                printf "%s_%s_%s", $(n-2), $(n-1), $n
            } else {
                printf "%s", $n
            }
        }')
    else
        # Middlebury: Lấy từ IMG_PATH như cũ
        IMG_PATH=$(dirname "$LEFT_IMAGE")
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
    fi
    
    if [ -z "$SCENE_NAME" ]; then
        SCENE_NAME=$(basename "$SCENE")
    fi
    
    OUTPUT_DIR_FULL="${OUTPUT_DIR}/${SCENE_NAME}"
    mkdir -p "$OUTPUT_DIR_FULL"
    
    echo "📁 Left:  $LEFT_IMAGE"
    echo "📁 Right: $RIGHT_IMAGE"
    echo "💾 Output: $OUTPUT_DIR_FULL"
    echo ""
    
    # Copy ảnh gốc vào output folder
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
        --mixed_precision 2>&1 | grep -E "(Processing|Error|Traceback)" || true
    
    EXIT_CODE=${PIPESTATUS[0]}
    cd ..
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Thành công!"
        SUCCESS=$((SUCCESS + 1))
    else
        echo "❌ Thất bại (exit code: $EXIT_CODE)"
        FAILED=$((FAILED + 1))
        FAILED_SCENES+=("$(basename $SCENE)")
    fi
    
    echo ""
done

# Báo cáo tổng kết
echo ""
echo "=========================================="
echo "📊 TỔNG KẾT"
echo "=========================================="
echo "✅ Thành công: $SUCCESS/${#SCENES[@]}"
echo "❌ Thất bại:   $FAILED/${#SCENES[@]}"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Danh sách scene thất bại:"
    for scene in "${FAILED_SCENES[@]}"; do
        echo "  - $scene"
    done
fi

echo ""
echo "📂 Kết quả:"
echo "   ls -lh $OUTPUT_DIR/"
echo ""
echo "💡 Xem tất cả ảnh disparity:"
echo "   eog $OUTPUT_DIR/*/im0_disp_jet.png"
echo "=========================================="

# =========================
# 🔧 CUSTOM CONFIGURATION (chỉ khi PRESET="custom")
# =========================
# Uncomment và chỉnh sửa phần này nếu bạn set PRESET="custom" ở đầu file

# CUSTOM_CONFIG() {
#     # Chọn dataset type
#     DATASET_TYPE="middlebury"  # hoặc "booster"
#     
#     # Option 1: Auto scan folder
#     AUTO_SCAN_DIR="datasets/your_folder"
#     SCENES=()
#     
#     # Option 2: Chọn scene cụ thể
#     # AUTO_SCAN_DIR=""
#     # SCENES=(
#     #     "datasets/mb2014/testH/Australia"
#     #     "datasets/mb2014/testH/Motorcycle"
#     #     "datasets/booster_gt/train/balanced/Kitchen"
#     # )
# }

