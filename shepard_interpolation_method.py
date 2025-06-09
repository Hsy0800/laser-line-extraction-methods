# Shepard插值方法用于亚像素激光线中心提取
# 参考: 基于Shepard插值的激光条纹中心提取方法

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class ShepardInterpolationMethod:
    """
    Shepard插值方法用于亚像素激光线中心提取
    
    特点:
    1. 使用Shepard插值进行亚像素精度提取
    2. 考虑局部邻域内的灰度分布
    3. 适应不同宽度和强度的激光线
    4. 对噪声具有较强的鲁棒性
    """
    
    def __init__(self, window_size=5, power_parameter=2.0, threshold=30):
        """
        初始化Shepard插值方法
        
        参数:
            window_size: 插值窗口大小，必须是奇数
            power_parameter: 权重计算中的幂参数，控制距离对权重的影响程度
            threshold: 灰度阈值，用于筛选激光线区域
        """
        if window_size % 2 == 0:
            window_size += 1  # 确保窗口大小为奇数
        
        self.window_size = window_size
        self.power_parameter = power_parameter
        self.threshold = threshold
        self.half_window = window_size // 2
    
    def _shepard_weight(self, distance, power_parameter):
        """
        计算Shepard插值的权重
        
        参数:
            distance: 距离
            power_parameter: 幂参数
            
        返回:
            weight: 权重值
        """
        if distance < 1e-10:
            return 1.0
        else:
            return 1.0 / (distance ** power_parameter)
    
    def _shepard_interpolate(self, image, x, y):
        """
        使用Shepard插值计算给定位置的灰度值
        
        参数:
            image: 输入图像
            x, y: 插值位置的坐标
            
        返回:
            value: 插值后的灰度值
        """
        # 确定插值窗口范围
        x_min = max(0, int(x) - self.half_window)
        x_max = min(image.shape[1] - 1, int(x) + self.half_window)
        y_min = max(0, int(y) - self.half_window)
        y_max = min(image.shape[0] - 1, int(y) + self.half_window)
        
        # 提取窗口内的像素坐标和灰度值
        total_weight = 0.0
        weighted_sum = 0.0
        
        for i in range(y_min, y_max + 1):
            for j in range(x_min, x_max + 1):
                # 计算距离
                distance = np.sqrt((j - x) ** 2 + (i - y) ** 2)
                
                # 计算权重
                weight = self._shepard_weight(distance, self.power_parameter)
                
                # 累加加权和
                weighted_sum += weight * image[i, j]
                total_weight += weight
        
        # 避免除以零
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0
    
    def _find_peak_position(self, profile):
        """
        在灰度剖面中找到峰值位置，使用抛物线拟合实现亚像素精度
        
        参数:
            profile: 灰度剖面
            
        返回:
            peak_pos: 峰值位置（亚像素精度）
        """
        if len(profile) < 3:
            return len(profile) // 2
        
        # 找到最大值位置
        max_idx = np.argmax(profile)
        
        # 边界检查
        if max_idx == 0 or max_idx == len(profile) - 1:
            return max_idx
        
        # 使用抛物线拟合实现亚像素精度
        y1, y2, y3 = profile[max_idx - 1], profile[max_idx], profile[max_idx + 1]
        
        # 避免除以零
        if 2 * (2 * y2 - y1 - y3) == 0:
            return max_idx
        
        # 计算亚像素峰值位置
        delta = 0.5 * (y1 - y3) / (y1 - 2 * y2 + y3)
        
        # 限制偏移量在合理范围内
        if abs(delta) > 1.0:
            delta = np.sign(delta) * 1.0
        
        return max_idx + delta
    
    def extract_centers(self, image):
        """
        提取图像中激光线的中心点
        
        参数:
            image: 输入的灰度图像
            
        返回:
            centers: 包含中心点坐标的列表，每个元素为(x, y)，表示亚像素精度的坐标
        """
        if len(image.shape) > 2:
            # 如果是彩色图像，转换为灰度图
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
        
        # 高斯平滑减少噪声
        smoothed = gaussian_filter(gray_image, sigma=1.0)
        
        # 自适应阈值分割
        binary = cv2.adaptiveThreshold(
            smoothed.astype(np.uint8),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            -2
        )
        
        # 形态学操作去除小噪点
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 细化为单像素宽度的线
        skeleton = cv2.ximgproc.thinning(binary)
        
        # 找到骨架点作为初始中心点
        y_indices, x_indices = np.where(skeleton > 0)
        initial_centers = list(zip(x_indices, y_indices))
        
        # 使用Shepard插值进行亚像素精度提取
        centers = []
        
        for x, y in initial_centers:
            # 估计局部方向
            direction = self._estimate_local_direction(skeleton, x, y)
            
            # 提取垂直于方向的灰度剖面
            profile, profile_coords = self._extract_perpendicular_profile(
                gray_image, x, y, direction)
            
            if len(profile) > 0:
                # 找到剖面中的峰值位置
                peak_idx = self._find_peak_position(profile)
                
                # 计算亚像素中心点坐标
                if 0 <= peak_idx < len(profile_coords):
                    # 线性插值获取亚像素坐标
                    idx_floor = int(np.floor(peak_idx))
                    idx_ceil = int(np.ceil(peak_idx))
                    
                    if idx_floor == idx_ceil:
                        sub_x, sub_y = profile_coords[idx_floor]
                    else:
                        alpha = peak_idx - idx_floor
                        x1, y1 = profile_coords[idx_floor]
                        x2, y2 = profile_coords[idx_ceil]
                        sub_x = (1 - alpha) * x1 + alpha * x2
                        sub_y = (1 - alpha) * y1 + alpha * y2
                    
                    # 使用Shepard插值进一步提高精度
                    intensity = self._shepard_interpolate(gray_image, sub_x, sub_y)
                    
                    # 如果插值强度足够高，添加到中心点列表
                    if intensity > self.threshold:
                        centers.append((sub_x, sub_y))
        
        return centers
    
    def _estimate_local_direction(self, skeleton, x, y, window_size=5):
        """
        估计局部方向
        
        参数:
            skeleton: 骨架图像
            x, y: 中心点坐标
            window_size: 局部窗口大小
            
        返回:
            angle: 局部方向角度（弧度）
        """
        # 提取局部窗口
        half_window = window_size // 2
        y_min = max(0, y - half_window)
        y_max = min(skeleton.shape[0], y + half_window + 1)
        x_min = max(0, x - half_window)
        x_max = min(skeleton.shape[1], x + half_window + 1)
        
        local_window = skeleton[y_min:y_max, x_min:x_max]
        
        # 找到局部窗口中的所有骨架点
        y_local, x_local = np.where(local_window > 0)
        
        # 如果点数太少，返回默认方向（水平）
        if len(y_local) < 2:
            return 0.0
        
        # 将局部坐标转换为全局坐标
        x_global = x_local + x_min
        y_global = y_local + y_min
        
        # 使用主成分分析估计方向
        x_mean = np.mean(x_global)
        y_mean = np.mean(y_global)
        x_centered = x_global - x_mean
        y_centered = y_global - y_mean
        
        # 计算协方差矩阵
        cov_xx = np.sum(x_centered * x_centered)
        cov_xy = np.sum(x_centered * y_centered)
        cov_yy = np.sum(y_centered * y_centered)
        
        # 避免除以零
        if abs(cov_xx - cov_yy) < 1e-10 and abs(cov_xy) < 1e-10:
            return 0.0
        
        # 计算主方向的角度
        if abs(cov_xy) < 1e-10:
            if cov_xx > cov_yy:
                angle = 0.0  # 水平方向
            else:
                angle = np.pi / 2  # 垂直方向
        else:
            # 计算协方差矩阵的特征值和特征向量
            theta = 0.5 * np.arctan2(2 * cov_xy, cov_xx - cov_yy)
            angle = theta
        
        return angle
    
    def _extract_perpendicular_profile(self, image, x, y, direction_angle, profile_length=7):
        """
        提取垂直于给定方向的灰度剖面
        
        参数:
            image: 输入图像
            x, y: 中心点坐标
            direction_angle: 方向角度（弧度）
            profile_length: 剖面长度
            
        返回:
            profile: 灰度剖面
            coords: 剖面上的坐标点
        """
        # 计算垂直方向
        perpendicular_angle = direction_angle + np.pi / 2
        
        # 计算垂直方向的单位向量
        dx = np.cos(perpendicular_angle)
        dy = np.sin(perpendicular_angle)
        
        # 提取剖面
        half_length = profile_length // 2
        profile = []
        coords = []
        
        for i in range(-half_length, half_length + 1):
            # 计算采样点坐标
            sample_x = x + i * dx
            sample_y = y + i * dy
            
            # 确保坐标在图像范围内
            if (0 <= int(sample_y) < image.shape[0] - 1 and 
                0 <= int(sample_x) < image.shape[1] - 1):
                # 使用双线性插值获取灰度值
                intensity = cv2.getRectSubPix(image, (1, 1), (sample_x, sample_y))[0, 0]
                profile.append(intensity)
                coords.append((sample_x, sample_y))
        
        return np.array(profile), coords
    
    def visualize_centers(self, image, centers):
        """
        可视化提取的中心点
        
        参数:
            image: 原始图像
            centers: 中心点列表
        """
        if len(image.shape) > 2:
            vis_image = image.copy()
        else:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 绘制中心点
        for x, y in centers:
            cv2.circle(vis_image, (int(x), int(y)), 1, (0, 0, 255), -1)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title('Extracted Laser Line Centers (Shepard Interpolation)')
        plt.axis('off')
        plt.show()

# 使用示例
def example_usage():
    # 创建一个模拟的激光线图像
    image = np.zeros((300, 400), dtype=np.uint8)
    
    # 添加一条弯曲的激光线
    for x in range(50, 350):
        y = int(150 + 50 * np.sin((x - 50) / 100))
        # 模拟激光线的高斯分布
        for dy in range(-5, 6):
            intensity = 255 * np.exp(-0.5 * (dy ** 2) / 4)
            if 0 <= y + dy < image.shape[0]:
                image[y + dy, x] = min(255, int(intensity))
    
    # 添加一些噪声
    noise = np.random.normal(0, 15, image.shape).astype(np.int32)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # 使用Shepard插值方法提取中心点
    shepard = ShepardInterpolationMethod(window_size=5, power_parameter=2.0, threshold=30)
    centers = shepard.extract_centers(image)
    
    # 可视化结果
    shepard.visualize_centers(image, centers)
    
    return image, centers

if __name__ == "__main__":
    example_usage()
