# Hướng Dẫn Sử Dụng Context-Aware MapReduce (Tích hợp trong Model)

## Tổng quan

Context-Aware MapReduce đã được **tích hợp sẵn vào mô hình MapReduceInference**. Bạn chỉ cần bật flag `--use_global_guidance` khi chạy test hoặc demo, mô hình sẽ tự động:
1. Chạy inference ở độ phân giải thấp (low-res pass)
2. Cache kết quả làm global guidance
3. Sử dụng guidance để cải thiện kết quả MapReduce

**Không cần chạy 2 script riêng biệt nữa!**

## Cách sử dụng nhanh

### 1. Với run_test_mapreduce_v2.py (Đơn giản nhất)

Mở file `run_test_mapreduce_v2.py` và chỉnh:

```python
# Dòng 41-43: Bật global guidance
USE_GLOBAL_GUIDANCE = True   # Bật tính năng
GUIDANCE_SCALE = 2.0        # Scale cho low-res pass
GUIDANCE_WEIGHT = 0.3       # Trọng số guidance (0-1)
```

Chạy:
```bash
python run_test_mapreduce_v2.py
```

### 2. Với command line trực tiếp

```bash
python test_mapreduce_v2.py \
    --datapath datasets/mb2014/trainingH \
    --dataset middlebury \
    --loadstereomodel pretrained/sceneflow.tar \
    --loadmonomodel pretrained/depth_anything_v2_vitl.pth \
    --iscale 1 \
    --oscale 1 \
    --tile_preset middlebury \
    --mixed_precision \
    --use_truncate_vol \
    --use_aggregate_mono_vol \
    --use_global_guidance \
    --guidance_scale 2.0 \
    --guidance_weight 0.3
```

## Cách hoạt động

### Workflow tự động trong model:

```
┌─────────────────────────────────────────────────────┐
│  User gọi inferencer.infer() với ảnh full-res      │
└──────────────────┬──────────────────────────────────┘
                   │
                   ├──> use_global_guidance = True?
                   │
        ┌──────────┴──────────┐
        │ YES                 │ NO
        │                     │
        ▼                     ▼
┌───────────────────┐  ┌─────────────────┐
│ 1. Downscale ảnh  │  │ Chạy MapReduce  │
│    theo scale     │  │ bình thường     │
├───────────────────┤  └─────────────────┘
│ 2. Chạy inference │
│    ở low-res      │
├───────────────────┤
│ 3. Upscale kết    │
│    quả → guidance │
├───────────────────┤
│ 4. Cache guidance │
└────────┬──────────┘
         │
         ▼
┌─────────────────────────────────┐
│ 5. Chạy MapReduce với guidance  │
│    - Mỗi tile được blend với    │
│      guidance tương ứng         │
│    - Confidence-based blending  │
└─────────────────────────────────┘
```

### Blending mechanism trong tile:

```python
# Trong _accumulate_tile của TileWrapper
diff = abs(prediction - guidance)
confidence = 1.0 - (diff / max_diff)
guidance_influence = guidance_weight * confidence
result = (1 - guidance_influence) * prediction + guidance_influence * guidance
```

## Tham số quan trọng

| Tham số | Giá trị khuyến nghị | Ý nghĩa |
|---------|---------------------|---------|
| `--use_global_guidance` | flag | Bật/tắt tính năng (không có giá trị) |
| `--guidance_scale` | 2.0 | Tỷ lệ downscale cho low-res pass |
| `--guidance_weight` | 0.3 | Mức độ ảnh hưởng của guidance (0-1) |

### Điều chỉnh theo trường hợp:

**Nếu kết quả không cải thiện:**
- Tăng `guidance_weight` lên 0.4-0.5
- Giảm `guidance_scale` xuống 1.5 (chất lượng guidance cao hơn)

**Nếu quá chậm:**
- Tăng `guidance_scale` lên 3.0 (low-res pass nhanh hơn)
- Model sẽ cache guidance, lần chạy thứ 2 trở đi nhanh hơn

**Nếu Out of Memory:**
- Tăng `guidance_scale` lên 3.0-4.0
- Dùng tile preset nhỏ hơn ('low_memory')

## So sánh với phương pháp cũ

| Khía cạnh | Phương pháp cũ (2-script) | Phương pháp mới (tích hợp) |
|-----------|---------------------------|----------------------------|
| **Số bước** | 3 bước thủ công | 1 bước tự động |
| **Storage** | Cần lưu guidance files | Không cần (cache trong RAM) |
| **Tốc độ** | Chậm (2 passes riêng) | Nhanh hơn (cache, tối ưu) |
| **Dễ sử dụng** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Linh hoạt** | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## Ví dụ thực tế

### Ví dụ 1: Middlebury với guidance

```python
# Trong run_test_mapreduce_v2.py
DATASET_NAME = 'middlebury'
TILE_PRESET = 'middlebury'
USE_GLOBAL_GUIDANCE = True
GUIDANCE_SCALE = 2.0
GUIDANCE_WEIGHT = 0.3
```

### Ví dụ 2: KITTI với guidance mạnh hơn

```python
DATASET_NAME = 'kitti2015'
TILE_PRESET = 'kitti'
USE_GLOBAL_GUIDANCE = True
GUIDANCE_SCALE = 1.5  # Chất lượng guidance cao hơn
GUIDANCE_WEIGHT = 0.4  # Ảnh hưởng mạnh hơn
```

### Ví dụ 3: GPU memory thấp

```python
DATASET_NAME = 'booster'
TILE_PRESET = 'low_memory'
USE_GLOBAL_GUIDANCE = True
GUIDANCE_SCALE = 3.0  # Low-res nhiều hơn để tiết kiệm VRAM
GUIDANCE_WEIGHT = 0.25
```

### Ví dụ 4: MapReduce thuần túy (không guidance)

```python
DATASET_NAME = 'sceneflow'
TILE_PRESET = 'sceneflow'
USE_GLOBAL_GUIDANCE = False  # Tắt guidance
```

## Troubleshooting

### Vấn đề: "Guidance không cải thiện kết quả"
**Giải pháp:**
- Tăng `GUIDANCE_WEIGHT` lên 0.4-0.5
- Giảm `GUIDANCE_SCALE` xuống 1.5
- Kiểm tra preset tile có phù hợp không

### Vấn đề: "Quá chậm"
**Giải pháp:**
- Model cache guidance sau lần đầu, nhưng nếu vẫn chậm:
- Tăng `GUIDANCE_SCALE` lên 2.5-3.0
- Cache sẽ tự động xóa khi clear_cache=True

### Vấn đề: "Out of Memory"
**Giải pháp:**
- Tăng `GUIDANCE_SCALE` lên 3.0-4.0 (guidance nhỏ hơn)
- Dùng `--clear_cache` để xóa cache sau mỗi sample
- Giảm tile size

### Vấn đề: "Kết quả có artifacts"
**Giải pháp:**
- Tăng overlap trong preset
- Điều chỉnh `GUIDANCE_WEIGHT` (thử cả tăng và giảm)
- Kiểm tra `GUIDANCE_SCALE` (2.0 thường tốt nhất)

## API cho developer

Nếu bạn muốn sử dụng trong code Python:

```python
from mapreduce_v2 import MapReduceInference

# Khởi tạo với guidance
inferencer = MapReduceInference(
    stereo_model,
    mono_model=None,
    tile_width=768,
    tile_height=512,
    overlap=96,
    mixed_precision=True,
    use_global_guidance=True,      # Bật guidance
    guidance_scale=2.0,             # Scale cho low-res
    guidance_weight=0.3,            # Trọng số
)

# Inference tự động sử dụng guidance
disparity = inferencer.infer(
    left_img,
    right_img,
    iscale=1.0,
    mono_pair=(mono_left, mono_right),
    verbose=True
)
```

## Ưu nhược điểm

### Ưu điểm:
✅ **Tự động hoàn toàn** - không cần chạy 2 script  
✅ **Cache thông minh** - lần chạy sau nhanh hơn  
✅ **Tích hợp sâu** - guidance được áp dụng ngay trong model  
✅ **Dễ sử dụng** - chỉ cần 1 flag  
✅ **Linh hoạt** - có thể bật/tắt dễ dàng  

### Nhược điểm:
❌ Vẫn chậm hơn pure downscale (do có thêm low-res pass)  
❌ Sử dụng thêm memory cho cache  
❌ Không thể tùy chỉnh guidance từ file external  

## Kết luận

Phương pháp mới giúp việc sử dụng Context-Aware MapReduce **đơn giản hơn nhiều** so với trước đây. Chỉ cần:

1. Bật `USE_GLOBAL_GUIDANCE = True` trong `run_test_mapreduce_v2.py`
2. Chạy `python run_test_mapreduce_v2.py`
3. Model tự động lo phần còn lại!

Phù hợp cho hầu hết trường hợp sử dụng. Nếu cần điều khiển chi tiết hơn (ví dụ: guidance từ file external), vẫn có thể dùng phương pháp cũ với `run_test_contextaware_mapreduce.py`.
