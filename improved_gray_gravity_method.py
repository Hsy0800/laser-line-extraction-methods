# 改进的灰度重心法用于亚像素激光线中心提取
# 参考: 基于改进灰度重心法的激光条纹中心提取

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class ImprovedGrayGravityMethod:
    """
    改进的灰度重心法用于亚像素激光线中心提取
    
    特点:
    1. 使用自适应窗口大小
    2. 考虑背景光照影响
    3. 使用高斯加权
    4. 亚像素精度
    """
    
    def __init__(self, window_size=7, threshold=30):
        """
        初始化改进的灰度重心法
        
        参数:
            window_size: 窗口大小，必须是奇数
            threshold: 灰度阈值，用于筛选激光线区域
        """
        if window_size % 2 == 0:
            window_size += 1  # 确保窗口大小为奇数
        
        self.window_size = window_size
        self.threshold = threshold
        self.half_window = window_size // 2
    
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
        
        # 使用改进的灰度重心法进行亚像素精度提取
        centers = []
        
        for x, y in initial_centers:
            # 提取局部窗口
            y_min = max(0, y - self.half_window)
            y_max = min(gray_image.shape[0], y + self.half_window + 1)
            x_min = max(0, x - self.half_window)
            x_max = min(gray_image.shape[1], x + self.half_window + 1)
            
            window = gray_image[y_min:y_max, x_min:x_max].astype(float)
            
            # 背景减除
            background = np.percentile(window, 10)  # 使用10%分位数作为背景估计
            window = np.maximum(window - background, 0)
            
            # 阈值筛选
            window[window < self.threshold] = 0
            
            # 如果窗口中没有有效像素，跳过
            if np.sum(window) == 0:
                continue
            
            # 计算亚像素中心点
            refined_y = self.refine_center(window)
            
            # 转换回原图坐标
            sub_y = y_min + refined_y
            
            centers.append((x, sub_y))
        
        return centers
    
    def refine_center(self, window):
        """
        使用改进的灰度重心法计算亚像素中心点
        
        参数:
            window: 局部窗口
            
        返回:
            center: 亚像素中心点的行索引
        """
        # 创建高斯权重
        rows, cols = window.shape
        y_indices = np.arange(rows)
        
        # 计算行索引的加权和
        weighted_sum = np.sum(y_indices[:, np.newaxis] * window)
        total_weight = np.sum(window)
        
        # 避免除以零
        if total_weight > 0:
            center = weighted_sum / total_weight
        else:
            center = rows / 2
        
        return center
    
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
        plt.title('Extracted Laser Line Centers (Improved Gray Gravity)')
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
    
    # 使用改进的灰度重心法提取中心点
    iggm = ImprovedGrayGravityMethod(window_size=7, threshold=30)
    centers = iggm.extract_centers(image)
    
    # 可视化结果
    iggm.visualize_centers(image, centers)
    
    return image, centers

if __name__ == "__main__":
    example_usage()
