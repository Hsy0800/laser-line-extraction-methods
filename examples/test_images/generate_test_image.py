#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成用于测试的激光线图像

此脚本可以生成带有噪声的模拟激光线图像，用于测试不同的激光线提取方法。
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def generate_laser_line_image(width=640, height=480, noise_level=0.1, line_width=5.0, 
                             line_intensity=200, background_intensity=50, save_path=None):
    """
    生成模拟的激光线图像
    
    参数:
        width (int): 图像宽度
        height (int): 图像高度
        noise_level (float): 噪声水平 (0-1)
        line_width (float): 激光线宽度
        line_intensity (int): 激光线强度 (0-255)
        background_intensity (int): 背景强度 (0-255)
        save_path (str): 保存路径，如果为None则不保存
        
    返回:
        numpy.ndarray: 生成的图像
    """
    # 创建背景
    image = np.ones((height, width), dtype=np.uint8) * background_intensity
    
    # 生成正弦曲线作为激光线
    x = np.arange(0, width)
    # 使用正弦函数生成曲线，可以调整参数改变曲线形状
    y = height // 2 + int(height * 0.2 * np.sin(x * 2 * np.pi / width))
    
    # 绘制激光线（高斯分布）
    for i in range(width):
        for j in range(height):
            # 计算到激光线中心的距离
            dist = abs(j - y[i])
            # 使用高斯函数模拟激光线强度分布
            intensity = line_intensity * np.exp(-(dist**2) / (2 * line_width**2))
            image[j, i] += int(intensity)
    
    # 确保像素值在有效范围内
    image = np.clip(image, 0, 255)
    
    # 添加高斯噪声
    noise = np.random.normal(0, noise_level * 255, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # 保存图像
    if save_path:
        plt.figure(figsize=(10, 8))
        plt.imshow(image, cmap='gray')
        plt.title('模拟激光线图像')
        plt.colorbar(label='强度')
        plt.savefig(save_path, dpi=100)
        plt.close()
        
        # 同时保存原始图像数据
        np.save(os.path.splitext(save_path)[0] + '.npy', image)
    
    return image


def generate_multiple_test_images(output_dir='./test_images', count=5):
    """
    生成多个测试图像，具有不同的参数
    
    参数:
        output_dir (str): 输出目录
        count (int): 要生成的图像数量
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成不同参数的图像
    for i in range(count):
        # 随机参数
        noise_level = 0.05 + 0.15 * np.random.random()  # 0.05-0.2
        line_width = 3.0 + 5.0 * np.random.random()     # 3.0-8.0
        line_intensity = 150 + 100 * np.random.random()  # 150-250
        background_intensity = 30 + 40 * np.random.random()  # 30-70
        
        # 生成图像
        save_path = os.path.join(output_dir, f'laser_line_test_{i+1}.png')
        generate_laser_line_image(
            noise_level=noise_level,
            line_width=line_width,
            line_intensity=line_intensity,
            background_intensity=background_intensity,
            save_path=save_path
        )
        
        print(f'已生成图像 {i+1}/{count}: {save_path}')


if __name__ == '__main__':
    # 生成单个测试图像
    image = generate_laser_line_image(save_path='laser_line_test.png')
    print('已生成测试图像: laser_line_test.png')
    
    # 生成多个测试图像
    generate_multiple_test_images(count=3)
    print('完成所有测试图像生成')
