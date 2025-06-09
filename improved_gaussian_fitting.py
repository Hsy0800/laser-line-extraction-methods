# 改进的高斯拟合方法用于亚像素激光线中心提取
# 参考: 基于改进高斯拟合的激光条纹中心提取方法

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

class ImprovedGaussianFitting:
    """
    改进的高斯拟合方法用于亚像素激光线中心提取
    
    特点:
    1. 使用标准高斯模型和非对称高斯模型进行拟合
    2. 迭代优化提高亚像素精度
    3. 自适应处理不同宽度和强度的激光线
    4. 对噪声和非均匀照明具有较强的鲁棒性
    """
    
    def __init__(self, window_size=7, threshold=30, max_iterations=5, convergence_threshold=0.1):
        """
        初始化改进的高斯拟合方法
        
        参数:
            window_size: 拟合窗口大小，必须是奇数
            threshold: 灰度阈值，用于筛选激光线区域
            max_iterations: 迭代优化的最大迭代次数
            convergence_threshold: 收敛阈值，当中心点位置变化小于此值时停止迭代
        """
        if window_size % 2 == 0:
            window_size += 1  # 确保窗口大小为奇数
        
        self.window_size = window_size
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.half_window = window_size // 2
    
    def _gaussian_model(self, x, amplitude, center, sigma, offset):
        """
        标准高斯模型
        
        参数:
            x: 自变量
            amplitude: 振幅
            center: 中心位置
            sigma: 标准差
            offset: 偏移量
            
        返回:
            y: 高斯函数值
        """
        return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + offset
    
    def _asymmetric_gaussian_model(self, x, amplitude, center, sigma_left, sigma_right, offset):
        """
        非对称高斯模型，左右两侧使用不同的标准差
        
        参数:
            x: 自变量
            amplitude: 振幅
            center: 中心位置
            sigma_left: 左侧标准差
            sigma_right: 右侧标准差
            offset: 偏移量
            
        返回:
            y: 非对称高斯函数值
        """
        result = np.zeros_like(x, dtype=float)
        
        # 左侧使用sigma_left
        left_mask = x <= center
        if np.any(left_mask):
            result[left_mask] = amplitude * np.exp(-0.5 * ((x[left_mask] - center) / sigma_left) ** 2) + offset
        
        # 右侧使用sigma_right
        right_mask = x > center
        if np.any(right_mask):
            result[right_mask] = amplitude * np.exp(-0.5 * ((x[right_mask] - center) / sigma_right) ** 2) + offset
        
        return result
    
    def _fit_gaussian(self, profile):
        """
        对灰度剖面进行高斯拟合
        
        参数:
            profile: 灰度剖面
            
        返回:
            center: 拟合得到的中心位置
            params: 拟合参数
        """
        if len(profile) < 3:
            return len(profile) // 2, None
        
        x = np.arange(len(profile))
        y = profile.astype(float)
        
        # 初始参数估计
        amplitude_init = np.max(y) - np.min(y)
        center_init = np.argmax(y)
        sigma_init = self.window_size / 6.0  # 经验值
        offset_init = np.min(y)
        
        # 参数边界
        bounds = (
            [0, 0, 0, 0],  # 下界
            [np.inf, len(profile), np.inf, np.inf]  # 上界
        )
        
        try:
            # 尝试标准高斯拟合
            params, _ = curve_fit(
                self._gaussian_model, 
                x, 
                y, 
                p0=[amplitude_init, center_init, sigma_init, offset_init],
                bounds=bounds,
                maxfev=1000
            )
            
            amplitude, center, sigma, offset = params
            return center, params
            
        except (RuntimeError, ValueError):
            try:
                # 如果标准高斯拟合失败，尝试非对称高斯拟合
                bounds_asymm = (
                    [0, 0, 0, 0, 0],  # 下界
                    [np.inf, len(profile), np.inf, np.inf, np.inf]  # 上界
                )
                
                params_asymm, _ = curve_fit(
                    self._asymmetric_gaussian_model, 
                    x, 
                    y, 
                    p0=[amplitude_init, center_init, sigma_init, sigma_init, offset_init],
                    bounds=bounds_asymm,
                    maxfev=1000
                )
                
                amplitude, center, sigma_left, sigma_right, offset = params_asymm
                return center, params_asymm
                
            except (RuntimeError, ValueError):
                # 如果两种拟合都失败，返回初始估计值
                return center_init, None
    
    def _estimate_local_direction(self, binary_image, x, y, window_size=7):
        """
        估计局部方向
        
        参数:
            binary_image: 二值图像
            x, y: 中心点坐标
            window_size: 局部窗口大小
            
        返回:
            angle: 局部方向角度（弧度）
        """
        # 提取局部窗口
        half_window = window_size // 2
        y_min = max(0, y - half_window)
        y_max = min(binary_image.shape[0], y + half_window + 1)
        x_min = max(0, x - half_window)
        x_max = min(binary_image.shape[1], x + half_window + 1)
        
        local_window = binary_image[y_min:y_max, x_min:x_max]
        
        # 找到局部窗口中的所有前景点
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
    
    def _extract_perpendicular_profile(self, image, x, y, direction_angle, profile_length=None):
        """
        提取垂直于给定方向的灰度剖面
        
        参数:
            image: 输入图像
            x, y: 中心点坐标
            direction_angle: 方向角度（弧度）
            profile_length: 剖面长度，如果为None则使用window_size
            
        返回:
            profile: 灰度剖面
            coords: 剖面上的坐标点
        """
        if profile_length is None:
            profile_length = self.window_size
        
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
    
    def _refine_center(self, image, x, y, direction_angle):
        """
        迭代优化中心点位置
        
        参数:
            image: 输入图像
            x, y: 初始中心点坐标
            direction_angle: 方向角度（弧度）
            
        返回:
            refined_x, refined_y: 优化后的中心点坐标
        """
        current_x, current_y = x, y
        
        for iteration in range(self.max_iterations):
            # 提取垂直于方向的灰度剖面
            profile, coords = self._extract_perpendicular_profile(
                image, current_x, current_y, direction_angle)
            
            if len(profile) < 3:
                break
            
            # 对剖面进行高斯拟合
            peak_idx, params = self._fit_gaussian(profile)
            
            if params is None:
                break
            
            # 计算亚像素中心点坐标
            if 0 <= peak_idx < len(coords):
                # 线性插值获取亚像素坐标
                idx_floor = int(np.floor(peak_idx))
                idx_ceil = int(np.ceil(peak_idx))
                
                if idx_floor == idx_ceil:
                    new_x, new_y = coords[idx_floor]
                else:
                    alpha = peak_idx - idx_floor
                    x1, y1 = coords[idx_floor]
                    x2, y2 = coords[idx_ceil]
                    new_x = (1 - alpha) * x1 + alpha * x2
                    new_y = (1 - alpha) * y1 + alpha * y2
                
                # 计算位置变化
                delta = np.sqrt((new_x - current_x) ** 2 + (new_y - current_y) ** 2)
                
                # 更新位置
                current_x, current_y = new_x, new_y
                
                # 检查收敛性
                if delta < self.convergence_threshold:
                    break
            else:
                break
        
        return current_x, current_y
    
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
        
        # 使用高斯拟合进行亚像素精度提取
        centers = []
        
        for x, y in initial_centers:
            # 估计局部方向
            direction = self._estimate_local_direction(skeleton, x, y)
            
            # 迭代优化中心点位置
            refined_x, refined_y = self._refine_center(gray_image, x, y, direction)
            
            # 检查优化后的中心点强度
            intensity = cv2.getRectSubPix(gray_image, (1, 1), (refined_x, refined_y))[0, 0]
            
            if intensity > self.threshold:
                centers.append((refined_x, refined_y))
        
        # 移除重复的中心点
        centers = self._remove_duplicate_centers(centers)
        
        return centers
    
    def _remove_duplicate_centers(self, centers, distance_threshold=1.5):
        """
        移除距离过近的重复中心点
        
        参数:
            centers: 中心点列表
            distance_threshold: 距离阈值，小于此值的点被视为重复点
            
        返回:
            filtered_centers: 过滤后的中心点列表
        """
        if not centers:
            return []
        
        # 按x坐标排序
        sorted_centers = sorted(centers, key=lambda p: p[0])
        
        filtered_centers = [sorted_centers[0]]
        
        for i in range(1, len(sorted_centers)):
            x1, y1 = sorted_centers[i]
            x0, y0 = filtered_centers[-1]
            
            # 计算距离
            distance = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            
            # 如果距离大于阈值，添加到过滤后的列表
            if distance > distance_threshold:
                filtered_centers.append((x1, y1))
        
        return filtered_centers
    
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
        plt.title('Extracted Laser Line Centers (Improved Gaussian Fitting)')
        plt.axis('off')
        plt.show()
    
    def visualize_fitting(self, image, x, y):
        """
        可视化特定点的剖面提取和高斯拟合过程
        
        参数:
            image: 输入图像
            x, y: 中心点坐标
        """
        if len(image.shape) > 2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            vis_image = image.copy()
        else:
            gray_image = image.copy()
            vis_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        
        # 估计局部方向
        # 首先需要获取二值图像和骨架
        smoothed = gaussian_filter(gray_image, sigma=1.0)
        binary = cv2.adaptiveThreshold(
            smoothed.astype(np.uint8),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            -2
        )
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        skeleton = cv2.ximgproc.thinning(binary)
        
        direction = self._estimate_local_direction(skeleton, x, y)
        
        # 提取垂直于方向的灰度剖面
        profile, coords = self._extract_perpendicular_profile(
            gray_image, x, y, direction, profile_length=15)  # 使用更长的剖面以便可视化
        
        # 对剖面进行高斯拟合
        peak_idx, params = self._fit_gaussian(profile)
        
        # 绘制剖面线
        if len(coords) >= 2:
            start_x, start_y = coords[0]
            end_x, end_y = coords[-1]
            cv2.line(vis_image, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 255, 0), 1)
        
        # 绘制原始中心点
        cv2.circle(vis_image, (int(x), int(y)), 3, (255, 0, 0), -1)
        
        # 如果拟合成功，绘制拟合后的中心点
        if params is not None and 0 <= peak_idx < len(coords):
            # 线性插值获取亚像素坐标
            idx_floor = int(np.floor(peak_idx))
            idx_ceil = int(np.ceil(peak_idx))
            
            if idx_floor == idx_ceil:
                refined_x, refined_y = coords[idx_floor]
            else:
                alpha = peak_idx - idx_floor
                x1, y1 = coords[idx_floor]
                x2, y2 = coords[idx_ceil]
                refined_x = (1 - alpha) * x1 + alpha * x2
                refined_y = (1 - alpha) * y1 + alpha * y2
            
            cv2.circle(vis_image, (int(refined_x), int(refined_y)), 3, (0, 0, 255), -1)
        
        # 创建图形
        plt.figure(figsize=(15, 6))
        
        # 绘制图像和剖面线
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title('Profile Extraction')
        plt.axis('off')
        
        # 绘制剖面和拟合曲线
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(len(profile)), profile, 'bo-', label='Original Profile')
        
        if params is not None:
            # 生成拟合曲线
            x_fit = np.linspace(0, len(profile) - 1, 100)
            if len(params) == 4:  # 标准高斯
                amplitude, center, sigma, offset = params
                y_fit = self._gaussian_model(x_fit, amplitude, center, sigma, offset)
                plt.plot(x_fit, y_fit, 'r-', label='Gaussian Fit')
                plt.axvline(x=center, color='r', linestyle='--', label='Fitted Center')
                plt.title(f'Gaussian Fitting (Center: {center:.2f})')
            elif len(params) == 5:  # 非对称高斯
                amplitude, center, sigma_left, sigma_right, offset = params
                y_fit = self._asymmetric_gaussian_model(x_fit, amplitude, center, sigma_left, sigma_right, offset)
                plt.plot(x_fit, y_fit, 'r-', label='Asymmetric Gaussian Fit')
                plt.axvline(x=center, color='r', linestyle='--', label='Fitted Center')
                plt.title(f'Asymmetric Gaussian Fitting (Center: {center:.2f})')
        else:
            plt.axvline(x=peak_idx, color='r', linestyle='--', label='Estimated Center')
            plt.title(f'Profile (Estimated Center: {peak_idx:.2f})')
        
        plt.xlabel('Position')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
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
    
    # 使用改进的高斯拟合方法提取中心点
    gaussian_fitting = ImprovedGaussianFitting(window_size=7, threshold=30)
    centers = gaussian_fitting.extract_centers(image)
    
    # 可视化结果
    gaussian_fitting.visualize_centers(image, centers)
    
    # 可视化特定点的拟合过程
    if centers:
        # 选择中间的一个点进行可视化
        sample_point = centers[len(centers) // 2]
        gaussian_fitting.visualize_fitting(image, sample_point[0], sample_point[1])
    
    return image, centers

if __name__ == "__main__":
    example_usage()
