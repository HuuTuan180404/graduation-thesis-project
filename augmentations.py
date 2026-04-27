import random
import numpy as np
from scipy.interpolate import interp1d

def add_gaussian_noise(data, mean=0.0, std=0.01):
    """Thêm nhiễu Gaussian vào dữ liệu keypoint."""
    noise = np.random.normal(mean, std, data.shape)
    return data + noise

def spatial_translate(data, max_offset=0.1):
    """Dịch chuyển toàn bộ bộ xương một cách nhất quán."""
    offset = np.random.uniform(-max_offset, max_offset, size=(1, 1, data.shape[2]))
    return data + offset

def spatial_scale(data, min_scale=0.8, max_scale=1.2):
    """Co giãn bộ xương từ tâm của nó."""
    scale_factor = random.uniform(min_scale, max_scale)
    # Tìm tâm của bộ xương ở khung hình đầu tiên để làm điểm neo
    center = np.mean(data[0, :, :], axis=0, keepdims=True)
    return (data - center) * scale_factor + center

def spatial_rotate(data, max_angle_deg=15):
    """Xoay bộ xương 2D quanh tâm của nó."""
    angle_rad = np.deg2rad(random.uniform(-max_angle_deg, max_angle_deg))
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    # Ma trận xoay 2D
    rotation_matrix = np.array([[c, -s], [s, c]])
    
    # Chỉ xoay tọa độ x, y (giả sử chúng là 2 chiều đầu tiên)
    xy_data = data[:, :, :2]
    
    # Tìm tâm để xoay
    center = np.mean(xy_data[0, :, :], axis=0, keepdims=True)
    
    # Áp dụng phép xoay
    rotated_xy = np.dot(xy_data - center, rotation_matrix.T) + center
    
    # Ghép lại với các chiều khác nếu có
    if data.shape[2] > 2:
        other_dims = data[:, :, 2:]
        return np.concatenate([rotated_xy, other_dims], axis=2)
    return rotated_xy

def temporal_warp(data, num_control_points=4, warp_factor=0.2):
    """Làm cong vênh chuỗi thời gian."""
    seq_len = data.shape[0]
    
    # Tạo các điểm điều khiển
    control_points_x = np.linspace(0, seq_len - 1, num_control_points)
    control_points_y = np.linspace(0, seq_len - 1, num_control_points)
    
    # Làm cong vênh các điểm điều khiển (trừ điểm đầu và cuối)
    random_offsets = np.random.uniform(-warp_factor, warp_factor, num_control_points - 2)
    control_points_y[1:-1] += random_offsets * seq_len
    
    # Tạo hàm nội suy
    warp_function = interp1d(control_points_x, control_points_y, kind='cubic')
    
    # Áp dụng hàm cong vênh
    warped_indices = warp_function(np.arange(seq_len))
    
    # Đảm bảo các chỉ số nằm trong giới hạn
    warped_indices[warped_indices < 0] = 0
    warped_indices[warped_indices >= seq_len] = seq_len - 1
    
    # Tạo dữ liệu mới
    new_data = np.zeros_like(data)
    for i in range(seq_len):
        # Nội suy tuyến tính giữa các khung hình gần nhất
        low_idx = int(np.floor(warped_indices[i]))
        high_idx = int(np.ceil(warped_indices[i]))
        interp_weight = warped_indices[i] - low_idx
        new_data[i] = (1 - interp_weight) * data[low_idx] + interp_weight * data[high_idx]
        
    return new_data

# Danh sách các hàm tăng cường dữ liệu để lựa chọn ngẫu nhiên
AVAILABLE_AUGMENTATIONS = [
    add_gaussian_noise,
    spatial_translate,
    spatial_scale,
    spatial_rotate,
    temporal_warp
]