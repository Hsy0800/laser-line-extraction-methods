# 多尺度各向异性高斯方法用于亚像素激光线中心提取
# 参考: 基于多尺度各向异性高斯核的激光条纹中心提取方法

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class MultiScaleAnisotropicGaussian:
    """
    多尺度各向异性高斯方法用于亚像素激光线中心提取
    
    特点:
    1. 使用多尺度处理，适应不同宽度的激光线
    2. 各向异性高斯核提高方向敏感性
    3. 基于Hessian矩阵的特征分析
    4. 高精度亚像素提取
    """
    
    def __init__(self, sigma_min=0.8, sigma_max=2.0, sigma_steps=3, threshold=10.0):
        """
        初始化多尺度各向异性高斯方法
        
        参数:
            sigma_min: 最小尺度
            sigma_max: 最大尺度
            sigma_steps: 尺度步数
            threshold: 响应阈值，用于筛选中心点
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_steps = sigma_steps
        self.threshold = threshold
    
    def _anisotropic_gaussian_kernel(self, size, sigma_x, sigma_y, theta):
        """
        生成各向异性高斯核
        
        参数:
            size: 核大小
            sigma_x: x方向的标准差
            sigma_y: y方向的标准差
            theta: 旋转角度（弧度）
            
        返回:
            kernel: 各向异性高斯核
        """
        # 确保size为奇数
        if size % 2 == 0:
            size += 1
        
        # 计算网格坐标
        half_size = size // 2
        x, y = np.meshgrid(np.arange(-half_size, half_size + 1), 
                           np.arange(-half_size, half_size + 1))
        
        # 坐标旋转
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_theta = x * cos_theta + y * sin_theta
        y_theta = -x * sin_theta + y * cos_theta
        
        # 计算高斯核
        kernel = np.exp(-(x_theta**2 / (2 * sigma_x**2) + y_theta**2 / (2 * sigma_y**2)))
        kernel = kernel / np.sum(kernel)  # 归一化
        
        return kernel
    
    def _compute_second_derivatives(self, image, sigma):
        """
        计算图像的二阶导数
        
        参数:
            image: 输入图像
            sigma: 高斯滤波的标准差
            
        返回:
            dxx, dxy, dyy: 二阶导数
        """
        # 使用高斯滤波平滑图像
        smoothed = gaussian_filter(image.astype(float), sigma)
        
        # 计算二阶导数
        dxx = cv2.Sobel(smoothed, cv2.CV_64F, 2, 0, ksize=3)
        dxy = cv2.Sobel(smoothed, cv2.CV_64F, 1, 1, ksize=3)
        dyy = cv2.Sobel(smoothed, cv2.CV_64F, 0, 2, ksize=3)
        
        return dxx, dxy, dyy
    
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
        
        # 多尺度处理
        sigmas = np.linspace(self.sigma_min, self.sigma_max, self.sigma_steps)
        
        # 存储所有尺度的候选点
        all_candidates = []
        all_responses = []
        
        for sigma in sigmas:
            # 计算二阶导数
            dxx, dxy, dyy = self._compute_second_derivatives(gray_image, sigma)
            
            # 构建Hessian矩阵并计算特征值
            candidates = []
            responses = []
            
            for y in range(1, gray_image.shape[0] - 1):
                for x in range(1, gray_image.shape[1] - 1):
                    # 构建Hessian矩阵
                    hessian = np.array([
                        [dxx[y, x], dxy[y, x]],
                        [dxy[y, x], dyy[y, x]]
                    ])
                    
                    # 计算特征值和特征向量
                    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
                    
                    # 找到绝对值最大的特征值及其对应的特征向量
                    idx = np.argmax(np.abs(eigenvalues))
                    max_eigenvalue = eigenvalues[idx]
                    max_eigenvector = eigenvectors[:, idx]
                    
                    # 检查是否为潜在的中心点（特征值足够大且为负）
                    if max_eigenvalue < -self.threshold:
                        # 计算垂直于线方向的二阶导数
                        nx, ny = max_eigenvector
                        
                        # 使用泰勒展开计算亚像素位置
                        dx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)[y, x]
                        dy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)[y, x]
                        
                        # 计算位移
                        t = -(dx * nx + dy * ny) / (dxx[y, x] * nx * nx + 2 * dxy[y, x] * nx * ny + dyy[y, x] * ny * ny)
                        
                        # 检查t是否在合理范围内
                        if abs(t) <= 1.0:
                            # 计算亚像素精度的中心点坐标
                            sub_x = x + t * nx
                            sub_y = y + t * ny
                            
                            # 确保坐标在图像范围内
                            if (0 <= sub_x < gray_image.shape[1] and 0 <= sub_y < gray_image.shape[0]):
                                candidates.append((sub_x, sub_y))
                                responses.append(abs(max_eigenvalue))
            
            all_candidates.extend(candidates)
            all_responses.extend(responses)
        
        # 非极大值抑制，去除重复点
        centers = self._non_maximum_suppression(all_candidates, all_responses, radius=2.0)
        
        return centers
    
    def _non_maximum_suppression(self, candidates, responses, radius=2.0):
        """
        非极大值抑制，去除重复点
        
        参数:
            candidates: 候选点列表
            responses: 对应的响应值
            radius: 抑制半径
            
        返回:
            centers: 抑制后的中心点列表
        """
        if not candidates:
            return []
        
        # 将候选点和响应值组合并按响应值排序
        points = list(zip(candidates, responses))
        points.sort(key=lambda x: x[1], reverse=True)
        
        # 非极大值抑制
        centers = []
        used = [False] * len(points)
        
        for i in range(len(points)):
            if used[i]:
                continue
            
            centers.append(points[i][0])  # 添加当前点
            used[i] = True
            
            # 抑制周围的点
            for j in range(i + 1, len(points)):
                if used[j]:
                    continue
                
                # 计算距离
                x1, y1 = points[i][0]
                x2, y2 = points[j][0]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                if distance < radius:
                    used[j] = True
        
        return centers
    
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
        plt.title('Extracted Laser Line Centers (Multi-Scale Anisotropic Gaussian)')
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
    
    # 使用多尺度各向异性高斯方法提取中心点
    msag = MultiScaleAnisotropicGaussian(sigma_min=0.8, sigma_max=2.0, sigma_steps=3, threshold=10.0)
    centers = msag.extract_centers(image)
    
    # 可视化结果
    msag.visualize_centers(image, centers)
    
    return image, centers

if __name__ == "__main__":
    example_usage()
