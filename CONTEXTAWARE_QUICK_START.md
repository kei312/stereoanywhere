# Quick Start: Context-Aware MapReduce

## Cách nhanh nhất để sử dụng

### 1. Chỉnh cấu hình (nếu cần)

Mở file `run_test_contextaware_mapreduce.py` và chỉnh:

```python
# Dòng 14-16: Thông tin dataset
DATAPATH = PROJECT_ROOT / 'datasets/mb2014/trainingH'
DATASET_NAME = 'middlebury'

# Dòng 21-24: Cấu hình processing
LOW_RES_SCALE = 2.0        # Tỷ lệ scale cho pass 1 (global context)
HIGH_RES_SCALE = 1.0       # Tỷ lệ scale cho pass 2 (MapReduce)
TILE_PRESET = 'middlebury' # Preset tile configuration
GUIDANCE_WEIGHT = 0.3      # Trọng số global guidance (0-1)
```

### 2. Chạy script

```bash
cd "/mnt/c/BTL PTDDL/stereoanywhere"
python run_test_contextaware_mapreduce.py
```

### 3. Xem kết quả

Kết quả được lưu trong: `output_contextaware/`

## Các tham số quan trọng

| Tham số | Giá trị khuyến nghị | Ý nghĩa |
|---------|---------------------|---------|
| LOW_RES_SCALE | 2.0 | Độ phân giải cho global context |
| HIGH_RES_SCALE | 1.0 | Độ phân giải cuối cùng |
| TILE_PRESET | Dataset-specific | Cấu hình tile (xem `tile_presets.py`) |
| GUIDANCE_WEIGHT | 0.3 | Mức độ ảnh hưởng của global guidance |

## Điều chỉnh theo trường hợp

### Nếu kết quả không tốt:
- Tăng `GUIDANCE_WEIGHT` lên 0.4-0.5
- Giảm `LOW_RES_SCALE` xuống 1.5

### Nếu quá chậm:
- Tăng `LOW_RES_SCALE` lên 3.0
- Dùng preset nhỏ hơn (ví dụ: 'low_memory')

### Nếu Out of Memory:
- Dùng `TILE_PRESET = 'low_memory'`
- Tăng `LOW_RES_SCALE` lên 3.0

## Xem chi tiết

Đọc `CONTEXTAWARE_MAPREDUCE_GUIDE.md` để biết thêm chi tiết về:
- Cách hoạt động của thuật toán
- So sánh với các phương pháp khác
- Troubleshooting
- Ví dụ thực tế cho nhiều dataset
