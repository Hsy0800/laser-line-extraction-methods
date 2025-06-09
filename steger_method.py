# Steger方法用于亚像素激光线中心提取
# 参考: Steger, C. (1998). An unbiased detector of curvilinear structures. IEEE Transactions on Pattern Analysis and Machine Intelligence, 20(2), 113-125.

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d

class StegerMethod:
    """
    Steger方法用于亚像素激光线中心提取
    
    特点:
    1. 基于二阶导数的零交叉点检测
    2. 使用Hessian矩阵的特征值和特征向量
    3. 高精度亚像素提取
    4. 对噪声具有较强的鲁棒性
    """
    
    def __init__(self, sigma=1.5, threshold=0.5):
        """
        初始化Steger方法
        
        参数:
            sigma: 高斯滤波的标准差
            threshold: 响应阈值，用于筛选中心点
        """
        self.sigma = sigma
        self.threshold = threshold
    
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
        
        # 高斯滤波
        smoothed = gaussian_filter(gray_image.astype(float), self.sigma)
        
        # 计算图像导数
        dx = gaussian_filter1d(smoothed, self.sigma, order=1, axis=1)
        dy = gaussian_filter1d(smoothed, self.sigma, order=1, axis=0)
        dxx = gaussian_filter1d(smoothed, self.sigma, order=2, axis=1)
        dxy = gaussian_filter1d(dx, self.sigma, order=1, axis=0)
        dyy = gaussian_filter1d(smoothed, self.sigma, order=2, axis=0)
        
        # 计算Hessian矩阵的特征值和特征向量
        centers = []
        
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
                    t = -(dx[y, x] * nx + dy[y, x] * ny) / (dxx[y, x] * nx * nx + 2 * dxy[y, x] * nx * ny + dyy[y, x] * ny * ny)
                    
                    # 检查t是否在合理范围内
                    if abs(t) <= 1.0:
                        # 计算亚像素精度的中心点坐标
                        sub_x = x + t * nx
                        sub_y = y + t * ny
                        
                        # 确保坐标在图像范围内
                        if (0 <= sub_x < gray_image.shape[1] and 0 <= sub_y < gray_image.shape[0]):
                            centers.append((sub_x, sub_y))
        
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
        plt.title('Extracted Laser Line Centers (Steger Method)')
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
    
    # 使用Steger方法提取中心点
    steger = StegerMethod(sigma=1.5, threshold=0.5)
    centers = steger.extract_centers(image)
    
    # 可视化结果
    steger.visualize_centers(image, centers)
    
    return image, centers

if __name__ == "__main__":
    example_usage()
