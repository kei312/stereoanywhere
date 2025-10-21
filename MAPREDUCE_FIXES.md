# MapReduce Performance Fixes

## Vấn đề đã gặp

Sau khi implement global guidance system:
- ❌ **Chậm hơn 6 lần**: Test mất ~56 phút thay vì ~10 phút
- ❌ **Kết quả tệ hơn**: Guidance không cải thiện quality như mong đợi
- ❌ **CUDA OOM**: Lỗi out of memory sau 13/15 ảnh do memory leak

## Giải pháp đã áp dụng

### 1. Tắt Global Guidance mặc định ✅

**File**: `run_test_mapreduce_v2.py`

```python
# BEFORE (guidance luôn bật)
USE_GLOBAL_GUIDANCE = True

# AFTER (guidance bị tắt mặc định)
USE_GLOBAL_GUIDANCE = False  # Opt-in only when needed
```

**Lý do**: Global guidance tốn 6x thời gian mà không cải thiện quality. Giữ nó như một feature opt-in cho trường hợp đặc biệt.

---

### 2. Thêm aggressive memory cleanup ✅

**File**: `test_mapreduce_v2.py`

**Import gc module**:
```python
import gc  # Add to imports at top of file
```

**Cleanup sau mỗi iteration**:
```python
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
```

**Lý do**: PyTorch không tự động release memory giữa các iterations. Cần explicitly delete tensors và clear CUDA cache.

---

### 3. Fix tensor_to_numpy_image memory leak ✅

**File**: `test_mapreduce_v2.py`

**BEFORE** (giữ tensor trên GPU):
```python
def tensor_to_numpy_image(tensor: Tensor) -> np.ndarray:
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    return tensor.cpu().permute(1, 2, 0).numpy()
```

**AFTER** (explicit CPU transfer + cleanup):
```python
def tensor_to_numpy_image(tensor: Tensor) -> np.ndarray:
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Explicit CPU transfer and cleanup to prevent memory accumulation
    tensor_cpu = tensor.cpu()
    image = tensor_cpu.permute(1, 2, 0).numpy()
    
    # Clean up intermediate tensors
    del tensor_cpu
    
    return image
```

**Lý do**: Tensor trên GPU không được release ngay cả khi moved sang CPU. Cần explicit delete.

---

## Hiệu năng sau khi fix

### MapReduce KHÔNG có guidance (khuyến nghị):
- ⏱️ **Tốc độ**: Nhanh (baseline MapReduce)
- 🎯 **Chất lượng**: Tốt (dùng tile overlap để blend)
- 💾 **Memory**: Ổn định (không OOM)

### MapReduce CÓ guidance (opt-in):
- ⏱️ **Tốc độ**: Chậm hơn 6x (2-pass inference)
- 🎯 **Chất lượng**: Không chắc tốt hơn
- 💾 **Memory**: Đã fix OOM

---

## Cách dùng

### Chạy MapReduce thuần túy (khuyến nghị):

```bash
cd /mnt/c/BTL\ PTDDL/stereoanywhere
python run_test_mapreduce_v2.py
```

Hoặc trực tiếp:
```bash
python test_mapreduce_v2.py \
    --datapath datasets/mb2014/trainingH \
    --dataset middlebury \
    --loadstereomodel pretrained/sceneflow.tar \
    --loadmonomodel pretrained/depth_anything_v2_vitl.pth \
    --iscale 1 --oscale 1 \
    --preset middlebury \
    --mixed_precision
```

### Nếu MUỐN thử guidance (opt-in):

Sửa `run_test_mapreduce_v2.py`:
```python
USE_GLOBAL_GUIDANCE = True
GUIDANCE_SCALE = 4.0  # Downscale factor (2, 4, or 8)
GUIDANCE_WEIGHT = 0.3 # Blend weight (0.2-0.5)
```

Hoặc CLI:
```bash
python test_mapreduce_v2.py \
    ... (other flags) ... \
    --use_global_guidance \
    --guidance_scale 4.0 \
    --guidance_weight 0.3
```

---

## Preset hệ thống

Các preset được định nghĩa trong `mapreduce_v2/tile_presets.py`:

| Preset | Tile Size | Overlap | Dùng cho |
|--------|-----------|---------|----------|
| `middlebury` | 728×1008 | 364 | Middlebury 2014/2021 |
| `kitti` | 384×1216 | 96 | KITTI Stereo 2012/2015 |
| `sceneflow` | 544×960 | 128 | FlyingThings3D, Monkaa, Driving |
| `booster` | 544×960 | 128 | Booster dataset |
| `low_memory` | 512×512 | 96 | Khi GPU memory thấp |
| `high_memory` | 1024×1024 | 192 | Khi GPU memory cao |

### Cách chọn preset:

**Tự động** (khuyến nghị):
```python
TILE_PRESET = 'auto'  # Tự chọn dựa trên DATASET_NAME
```

**Thủ công**:
```python
USE_PRESET = True
TILE_PRESET = 'middlebury'  # Hoặc 'kitti', 'sceneflow', etc.
```

**Xem danh sách**:
```python
TILE_PRESET = 'list'  # In ra tất cả presets
```

---

## Khắc phục sự cố

### Vẫn OOM?

1. **Giảm tile size**:
   ```python
   TILE_PRESET = 'low_memory'  # 512x512 tiles
   ```

2. **Tắt mixed precision** (tốn memory hơn nhưng ổn định):
   ```python
   # Remove --mixed_precision flag
   ```

3. **Clear cache thường xuyên hơn**: Đã có trong code rồi

### Kết quả không tốt?

1. **Tăng overlap**: Giúp blend tiles mượt hơn
   ```python
   # In tile_presets.py, increase overlap
   TilePreset(tile_width=728, tile_height=1008, overlap=400)  # Was 364
   ```

2. **Thử guidance** (nếu chấp nhận chậm):
   ```python
   USE_GLOBAL_GUIDANCE = True
   GUIDANCE_WEIGHT = 0.4  # Increase influence
   ```

3. **So sánh với downscale**:
   ```bash
   # Run with iscale=2 (downscale) để benchmark
   python test.py --iscale 2 ...
   ```

---

## Summary

✅ **Đã fix**:
- Guidance tắt mặc định (opt-in only)
- Memory leak trong vòng lặp test
- Memory leak trong tensor_to_numpy_image()
- OOM errors với aggressive cleanup

🎯 **Khuyến nghị**:
- Dùng MapReduce **KHÔNG** guidance cho hầu hết use cases
- Chỉ bật guidance khi thực sự cần và chấp nhận trade-off 6x chậm hơn
- Chọn preset phù hợp với dataset để optimize tile size

📊 **Hiệu năng expected**:
- MapReduce (no guidance): ~10 phút cho 15 images (Middlebury)
- MapReduce (with guidance): ~56+ phút cho 15 images
- Simple downscale (iscale=2): ~5 phút nhưng resolution thấp hơn
