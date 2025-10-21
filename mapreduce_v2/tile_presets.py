"""
Tile Configuration Presets for Different Datasets
Định nghĩa các preset cấu hình tile cho các bộ dữ liệu khác nhau.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TilePreset:
    """Cấu hình tile cho một preset cụ thể."""
    name: str
    tile_width: int
    tile_height: int
    overlap: int
    description: str = ""
    
    def __post_init__(self):
        """Validate các giá trị."""
        if self.tile_width <= 0 or self.tile_height <= 0:
            raise ValueError(f"tile_width và tile_height phải > 0, nhận được {self.tile_width}x{self.tile_height}")
        if self.overlap < 0:
            raise ValueError(f"overlap phải >= 0, nhận được {self.overlap}")
        # Khuyến nghị: tile size nên chia hết cho 32 (cho CNN)
        if self.tile_width % 32 != 0:
            print(f"WARNING: tile_width={self.tile_width} không chia hết cho 32, có thể gây vấn đề padding")
        if self.tile_height % 32 != 0:
            print(f"WARNING: tile_height={self.tile_height} không chia hết cho 32, có thể gây vấn đề padding")


# ==============================================================================
# ĐỊNH NGHĨA CÁC PRESET
# Chỉnh sửa các giá trị ở đây để thay đổi cấu hình cho từng dataset
# ==============================================================================

TILE_PRESETS = {
    # Preset mặc định - tile vuông cân bằng
    "default": TilePreset(
        name="default",
        tile_width=448,
        tile_height=448,
        overlap=96,
        description="Tile vuông mặc định, phù hợp cho hầu hết trường hợp"
    ),
    
    # Middlebury dataset - ảnh có độ phân giải trung bình đến cao
    "middlebury": TilePreset(
        name="middlebury",
        tile_width=672 ,
        tile_height=1120,
        overlap=112,
        description="Tối ưu cho Middlebury dataset (ảnh ~1300x700-1000)"
    ),
    
    # KITTI dataset - ảnh wide aspect ratio (1242x375 hoặc tương tự)
    "kitti": TilePreset(
        name="kitti",
        tile_width=1344,
        tile_height=448,
        overlap=128,
        description="Tối ưu cho KITTI dataset (ảnh wide 1242x375)"
    ),
    
    # SceneFlow dataset - ảnh lớn (960x540 hoặc cao hơn)
    "sceneflow": TilePreset(
        name="sceneflow",
        tile_width=448,
        tile_height=448,
        overlap=112,
        description="Tối ưu cho SceneFlow dataset"
    ),
    
    # Booster dataset - ảnh có thể rất lớn
    "booster": TilePreset(
        name="booster",
        tile_width=1120,
        tile_height=896,
        overlap=224,
        description="Tối ưu cho Booster dataset (ảnh độ phân giải cao)"
    ),
    
    # MonoTrap dataset
    "monotrap": TilePreset(
        name="monotrap",
        tile_width=800,
        tile_height=600,
        overlap=96,
        description="Tối ưu cho MonoTrap dataset"
    ),
    
    # Preset cho ảnh nhỏ - tile lớn hơn để giảm số lượng tile
    "small_image": TilePreset(
        name="small_image",
        tile_width=1024,
        tile_height=1024,
        overlap=64,
        description="Cho ảnh nhỏ (<1024x1024), sử dụng tile lớn"
    ),
    
    # Preset cho ảnh rất lớn - tile nhỏ hơn để tiết kiệm VRAM
    "large_image": TilePreset(
        name="large_image",
        tile_width=512,
        tile_height=512,
        overlap=64,
        description="Cho ảnh rất lớn (>2048x2048), tiết kiệm VRAM"
    ),
    
    # Preset cho GPU memory thấp
    "low_memory": TilePreset(
        name="low_memory",
        tile_width=512,
        tile_height=384,
        overlap=48,
        description="Tiết kiệm VRAM tối đa, phù hợp GPU <8GB"
    ),
    
    # Preset cho GPU memory cao
    "high_memory": TilePreset(
        name="high_memory",
        tile_width=1280,
        tile_height=960,
        overlap=128,
        description="Sử dụng tile lớn, cần GPU >16GB"
    ),
}


# ==============================================================================
# HÀM TIỆN ÍCH
# ==============================================================================

def get_preset(preset_name: str) -> TilePreset:
    """
    Lấy preset theo tên.
    
    Args:
        preset_name: Tên preset (ví dụ: 'middlebury', 'kitti', 'default')
        
    Returns:
        TilePreset object
        
    Raises:
        ValueError: Nếu preset không tồn tại
    """
    if preset_name not in TILE_PRESETS:
        available = ", ".join(TILE_PRESETS.keys())
        raise ValueError(
            f"Preset '{preset_name}' không tồn tại. "
            f"Các preset khả dụng: {available}"
        )
    return TILE_PRESETS[preset_name]


def list_presets() -> None:
    """In danh sách tất cả các preset có sẵn."""
    print("Các Tile Preset có sẵn:")
    print("=" * 80)
    for name, preset in TILE_PRESETS.items():
        print(f"\n[{name}]")
        print(f"  Tile size: {preset.tile_width}x{preset.tile_height}")
        print(f"  Overlap: {preset.overlap}px")
        print(f"  Mô tả: {preset.description}")
    print("=" * 80)


def get_preset_for_dataset(dataset_name: str) -> TilePreset:
    """
    Tự động chọn preset phù hợp dựa trên tên dataset.
    
    Args:
        dataset_name: Tên dataset (ví dụ: 'middlebury', 'kitti2015', 'sceneflow', etc.)
        
    Returns:
        TilePreset object phù hợp
    """
    dataset_lower = dataset_name.lower()
    
    # Mapping từ tên dataset sang preset
    dataset_mapping = {
        'middlebury': 'middlebury',
        'middlebury2014': 'middlebury',
        'middlebury2021': 'middlebury',
        'kitti': 'kitti',
        'kitti2012': 'kitti',
        'kitti2015': 'kitti',
        'sceneflow': 'sceneflow',
        'flyingthings': 'sceneflow',
        'driving': 'sceneflow',
        'monkaa': 'sceneflow',
        'booster': 'booster',
        'monotrap': 'monotrap',
    }
    
    # Tìm preset phù hợp
    for key, preset_name in dataset_mapping.items():
        if key in dataset_lower:
            print(f"Auto-selected preset '{preset_name}' for dataset '{dataset_name}'")
            return TILE_PRESETS[preset_name]
    
    # Fallback về default
    print(f"No specific preset for dataset '{dataset_name}', using 'default'")
    return TILE_PRESETS['default']


def create_custom_preset(
    name: str,
    tile_width: int,
    tile_height: int,
    overlap: int,
    description: str = ""
) -> TilePreset:
    """
    Tạo preset tùy chỉnh tạm thời (không lưu vào TILE_PRESETS).
    
    Args:
        name: Tên preset
        tile_width: Chiều rộng tile
        tile_height: Chiều cao tile
        overlap: Độ chồng lấp
        description: Mô tả preset
        
    Returns:
        TilePreset object mới
    """
    return TilePreset(
        name=name,
        tile_width=tile_width,
        tile_height=tile_height,
        overlap=overlap,
        description=description
    )


# ==============================================================================
# MAIN - Để test
# ==============================================================================

if __name__ == "__main__":
    # In danh sách tất cả preset
    list_presets()
    
    # Test lấy preset
    print("\n\nTest lấy preset:")
    preset = get_preset("middlebury")
    print(f"Preset 'middlebury': {preset.tile_width}x{preset.tile_height}, overlap={preset.overlap}")
    
    # Test auto-select dựa trên dataset
    print("\n\nTest auto-select:")
    for dataset in ['middlebury2014', 'kitti2015', 'sceneflow', 'unknown_dataset']:
        preset = get_preset_for_dataset(dataset)
        print(f"  {dataset} -> {preset.name}")
