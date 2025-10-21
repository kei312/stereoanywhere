# Hướng Dẫn Sử Dụng Context-Aware MapReduce

## Tổng quan

Context-Aware MapReduce là giải pháp cải thiện chất lượng output của MapReduce bằng cách sử dụng **global guidance** từ một lần inference ở độ phân giải thấp hơn. Phương pháp này giải quyết vấn đề mất nhất quán toàn cục khi xử lý các tile độc lập.

## Cách hoạt động

### Two-Pass Processing:

1. **Pass 1 (Low-Resolution)**: 
   - Chạy mô hình ở độ phân giải thấp (iscale=2.0)
   - Tạo disparity map với ngữ cảnh toàn cục
   - Lưu kết quả làm "global guidance"

2. **Pass 2 (High-Resolution with Guidance)**:
   - Chạy MapReduce ở độ phân giải đầy đủ (iscale=1.0)
   - Sử dụng global guidance để điều chỉnh kết quả từng tile
   - Blend giữa prediction và guidance dựa trên confidence

### Cơ chế blending:

```python
# Tính confidence dựa trên sự khác biệt
diff = abs(prediction - guidance)
confidence = 1.0 - (diff / max_diff)

# Blend với trọng số động
guidance_influence = guidance_weight * confidence
result = (1 - guidance_influence) * prediction + guidance_influence * guidance
```

## Cách sử dụng

### Cách 1: Sử dụng script tự động (Khuyến nghị)

```bash
cd /mnt/c/BTL\ PTDDL/stereoanywhere
python run_test_contextaware_mapreduce.py
```

Script này sẽ tự động:
- Chạy Pass 1 (low-res)
- Tạo guidance maps
- Chạy Pass 2 (high-res với guidance)

### Cách 2: Chạy thủ công từng bước

**Bước 1: Chạy low-res pass**
```bash
python test.py \
    --datapath datasets/mb2014/trainingH \
    --dataset middlebury \
    --loadstereomodel pretrained/sceneflow.tar \
    --loadmonomodel pretrained/depth_anything_v2_vitl.pth \
    --iscale 2 \
    --oscale 2 \
    --outdir output_contextaware/temp/lowres \
    --mixed_precision \
    --use_truncate_vol \
    --use_aggregate_mono_vol
```

**Bước 2: Tạo guidance maps**
```python
import numpy as np
from pathlib import Path
import cv2

lowres_dir = Path('output_contextaware/temp/lowres/raw')
guidance_dir = Path('output_contextaware/temp/guidance')
guidance_dir.mkdir(parents=True, exist_ok=True)

for npy_file in lowres_dir.glob('*.npy'):
    disp = np.load(npy_file).squeeze()
    # Upscale x2
    h, w = disp.shape
    disp_upscaled = cv2.resize(disp, (w*2, h*2), interpolation=cv2.INTER_LINEAR)
    disp_upscaled *= 2.0  # Scale disparity values
    
    # Save
    guidance_file = guidance_dir / f'{npy_file.stem}_guidance.npy'
    np.save(guidance_file, disp_upscaled)
```

**Bước 3: Chạy high-res với guidance**
```bash
python test_mapreduce_v2.py \
    --datapath datasets/mb2014/trainingH \
    --dataset middlebury \
    --loadstereomodel pretrained/sceneflow.tar \
    --loadmonomodel pretrained/depth_anything_v2_vitl.pth \
    --iscale 1 \
    --oscale 1 \
    --tile_preset middlebury \
    --outdir output_contextaware \
    --mixed_precision \
    --use_truncate_vol \
    --use_aggregate_mono_vol \
    --guidance_dir output_contextaware/temp/guidance \
    --guidance_weight 0.3
```

## Tùy chỉnh tham số

### Trong `run_test_contextaware_mapreduce.py`:

```python
# Tỷ lệ scale cho pass 1
LOW_RES_SCALE = 2.0  # Cao hơn = nhanh hơn nhưng guidance kém chất lượng

# Tỷ lệ scale cho pass 2
HIGH_RES_SCALE = 1.0  # Thường giữ nguyên 1.0

# Preset tile cho MapReduce
TILE_PRESET = 'middlebury'  # Hoặc 'kitti', 'sceneflow', etc.

# Trọng số của global guidance
GUIDANCE_WEIGHT = 0.3  # 0-1, cao hơn = ảnh hưởng nhiều hơn từ guidance
```

### Ý nghĩa tham số:

- **LOW_RES_SCALE**: 
  - 2.0 = cân bằng tốt giữa tốc độ và chất lượng
  - 1.5 = guidance chất lượng cao hơn nhưng chậm hơn
  - 3.0 = nhanh nhưng guidance kém hơn

- **GUIDANCE_WEIGHT**:
  - 0.1-0.2 = ảnh hưởng nhẹ, giữ gần prediction gốc
  - 0.3-0.4 = cân bằng tốt (khuyến nghị)
  - 0.5-0.7 = ảnh hưởng mạnh, gần guidance hơn
  - >0.7 = quá phụ thuộc guidance, có thể mất chi tiết

## So sánh với các phương pháp khác

| Phương pháp | Tốc độ | Chất lượng | VRAM | Nhất quán |
|-------------|--------|------------|------|-----------|
| Pure Downscale (iscale=2) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Pure MapReduce (iscale=1) | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐ |
| **Context-Aware MapReduce** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| Hybrid (iscale=1.5 + MapReduce) | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

## Ưu nhược điểm

### Ưu điểm:
✅ Cải thiện đáng kể chất lượng so với pure MapReduce  
✅ Giữ được chi tiết ở full resolution  
✅ Giảm artifacts tại biên tile  
✅ Tăng tính nhất quán toàn cục  
✅ Không cần chỉnh sửa mô hình gốc  

### Nhược điểm:
❌ Chậm hơn pure downscale (cần 2 passes)  
❌ Yêu cầu lưu trữ tạm thời cho guidance maps  
❌ Phức tạp hơn trong cài đặt  
❌ Vẫn có thể có một số artifacts nhỏ  

## Troubleshooting

### Vấn đề: "No guidance maps created"
- Kiểm tra xem Pass 1 đã chạy thành công chưa
- Xem thư mục `output_contextaware/temp/lowres/raw` có file .npy không

### Vấn đề: "Failed to load guidance"
- Kiểm tra format file guidance (.npy)
- Đảm bảo guidance shape tương thích với ảnh input

### Vấn đề: Kết quả không cải thiện
- Thử tăng `GUIDANCE_WEIGHT` lên 0.4-0.5
- Thử giảm `LOW_RES_SCALE` xuống 1.5
- Kiểm tra preset tile có phù hợp với dataset không

### Vấn đề: Out of Memory
- Giảm tile size trong preset
- Sử dụng preset 'low_memory'
- Bật `--clear_cache`

## Ví dụ thực tế

### Ví dụ 1: Middlebury dataset
```python
# Trong run_test_contextaware_mapreduce.py
DATAPATH = PROJECT_ROOT / 'datasets/mb2014/trainingH'
DATASET_NAME = 'middlebury'
LOW_RES_SCALE = 2.0
TILE_PRESET = 'middlebury'
GUIDANCE_WEIGHT = 0.3
```

### Ví dụ 2: KITTI dataset
```python
DATAPATH = PROJECT_ROOT / 'datasets/kitti2015'
DATASET_NAME = 'kitti2015'
LOW_RES_SCALE = 2.0
TILE_PRESET = 'kitti'
GUIDANCE_WEIGHT = 0.35
```

### Ví dụ 3: GPU memory thấp
```python
DATAPATH = PROJECT_ROOT / 'datasets/booster_gt/train'
DATASET_NAME = 'booster'
LOW_RES_SCALE = 3.0  # Giảm chất lượng guidance để tiết kiệm
TILE_PRESET = 'low_memory'
GUIDANCE_WEIGHT = 0.25
```

## Kết luận

Context-Aware MapReduce là giải pháp tốt khi:
- Bạn cần xử lý ảnh độ phân giải cao
- Chất lượng quan trọng hơn tốc độ
- Pure MapReduce cho kết quả không tốt
- Bạn có đủ thời gian để chạy 2 passes

Không nên dùng khi:
- Tốc độ là ưu tiên hàng đầu
- Ảnh đủ nhỏ để chạy trực tiếp ở iscale=2
- Không có đủ storage cho guidance maps
