# 激光线中心提取方法基本使用示例

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加父目录到系统路径，以便导入模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入各种激光线中心提取方法
from improved_gray_gravity_method import ImprovedGrayGravityMethod
from steger_method import StegerMethod
from multi_scale_anisotropic_gaussian import MultiScaleAnisotropicGaussian
from shepard_interpolation_method import ShepardInterpolationMethod
from improved_gaussian_fitting import ImprovedGaussianFitting

# 可选：如果有预训练模型，导入深度学习方法
try:
    from deep_learning_laser_center import DeepLearningLaserCenter
    has_deep_learning = True
except ImportError:
    has_deep_learning = False

# 生成模拟的激光线图像
def generate_test_image(width=400, height=300, noise_level=10):
    # 创建空白图像
    image = np.zeros((height, width), dtype=np.uint8)
    
    # 生成激光线路径（正弦曲线）
    x = np.arange(0, width)
    y = height // 2 + 30 * np.sin(x * 2 * np.pi / width)
    y = y.astype(int)
    
    # 绘制激光线（高斯分布）
    for i, (xi, yi) in enumerate(zip(x, y)):
        if 0 <= xi < width and 0 <= yi < height:
            # 激光线宽度随x变化
            sigma = 2 + 1.5 * np.sin(xi * 4 * np.pi / width)
            for r in range(-15, 16):
                if 0 <= yi + r < height:
                    intensity = 200 * np.exp(-0.5 * (r / sigma) ** 2)
                    image[yi + r, xi] = min(255, int(intensity))
    
    # 添加噪声
    noise = np.random.normal(0, noise_level, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 返回图像和真实中心点
    centers = [(int(xi), int(yi)) for xi, yi in zip(x, y) if 0 <= xi < width and 0 <= yi < height]
    return image, centers

# 主函数
def main():
    # 生成测试图像
    print("生成测试激光线图像...")
    image, ground_truth = generate_test_image(noise_level=15)
    
    # 显示原始图像
    plt.figure(figsize=(10, 8))
    plt.subplot(231)
    plt.imshow(image, cmap='gray')
    plt.title('原始激光线图像')
    
    # 使用改进的灰度重心法
    print("使用改进的灰度重心法提取中心点...")
    iggm = ImprovedGrayGravityMethod(window_size=7, threshold=30)
    iggm_centers = iggm.extract_centers(image)
    
    plt.subplot(232)
    iggm.visualize_centers(image, iggm_centers)
    plt.title('改进的灰度重心法')
    
    # 使用Steger方法
    print("使用Steger方法提取中心点...")
    steger = StegerMethod(sigma=1.5, threshold=5.0)
    steger_centers = steger.extract_centers(image)
    
    plt.subplot(233)
    steger.visualize_centers(image, steger_centers)
    plt.title('Steger方法')
    
    # 使用多尺度各向异性高斯核方法
    print("使用多尺度各向异性高斯核方法提取中心点...")
    msag = MultiScaleAnisotropicGaussian(scales=[1.0, 1.5, 2.0], threshold=30)
    msag_centers = msag.extract_centers(image)
    
    plt.subplot(234)
    msag.visualize_centers(image, msag_centers)
    plt.title('多尺度各向异性高斯核')
    
    # 使用Shepard插值方法
    print("使用Shepard插值方法提取中心点...")
    sim = ShepardInterpolationMethod(window_size=7, threshold=30)
    sim_centers = sim.extract_centers(image)
    
    plt.subplot(235)
    sim.visualize_centers(image, sim_centers)
    plt.title('Shepard插值方法')
    
    # 使用改进的高斯曲线拟合方法
    print("使用改进的高斯曲线拟合方法提取中心点...")
    igf = ImprovedGaussianFitting(window_size=7, threshold=30)
    igf_centers = igf.extract_centers(image)
    
    plt.subplot(236)
    igf.visualize_centers(image, igf_centers)
    plt.title('改进的高斯曲线拟合')
    
    plt.tight_layout()
    plt.savefig('laser_line_extraction_comparison.png', dpi=300)
    plt.show()
    
    print("所有方法执行完成，结果已保存为图像。")

if __name__ == "__main__":
    main()