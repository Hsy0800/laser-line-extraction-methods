#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高级激光线测试图像生成器

此脚本可以生成多种类型的模拟激光线图像，包括：
- 直线型激光线
- 曲线型激光线
- 多线条激光线
- 不同噪声水平和对比度的图像

适用于全面测试不同激光线提取算法的性能。
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from enum import Enum
import argparse


class LaserLineType(Enum):
    """激光线类型枚举"""
    STRAIGHT = 'straight'  # 直线
    SINE = 'sine'          # 正弦曲线
    MULTIPLE = 'multiple'  # 多线条
    CROSS = 'cross'        # 交叉线条


def generate_advanced_laser_image(width=640, height=480, line_type=LaserLineType.SINE,
                                 noise_level=0.1, line_width=5.0, line_intensity=200,
                                 background_intensity=50, contrast=1.0, save_path=None):
    """
    生成高级模拟激光线图像
    
    参数:
        width (int): 图像宽度
        height (int): 图像高度
        line_type (LaserLineType): 激光线类型
        noise_level (float): 噪声水平 (0-1)
        line_width (float): 激光线宽度
        line_intensity (int): 激光线强度 (0-255)
        background_intensity (int): 背景强度 (0-255)
        contrast (float): 对比度调整因子
        save_path (str): 保存路径，如果为None则不保存
        
    返回:
        numpy.ndarray: 生成的图像
    """
    # 创建背景
    image = np.ones((height, width), dtype=np.float32) * background_intensity
    
    # 根据线型生成激光线
    if line_type == LaserLineType.STRAIGHT:
        # 直线型激光线
        y_center = height // 2
        # 添加一点随机倾斜
        slope = np.random.uniform(-0.2, 0.2)
        y_positions = [int(y_center + slope * x) for x in range(width)]
        
        for i in range(width):
            y_pos = y_positions[i]
            if 0 <= y_pos < height:
                for j in range(height):
                    dist = abs(j - y_pos)
                    intensity = line_intensity * np.exp(-(dist**2) / (2 * line_width**2))
                    image[j, i] += intensity
    
    elif line_type == LaserLineType.SINE:
        # 正弦曲线型激光线
        x = np.arange(0, width)
        # 随机调整正弦波参数
        amplitude = int(height * np.random.uniform(0.1, 0.3))
        frequency = np.random.uniform(1, 3) * 2 * np.pi / width
        phase = np.random.uniform(0, 2*np.pi)
        
        y = height // 2 + amplitude * np.sin(frequency * x + phase)
        y = y.astype(int)
        
        for i in range(width):
            y_pos = y[i]
            if 0 <= y_pos < height:
                for j in range(height):
                    dist = abs(j - y_pos)
                    intensity = line_intensity * np.exp(-(dist**2) / (2 * line_width**2))
                    image[j, i] += intensity
    
    elif line_type == LaserLineType.MULTIPLE:
        # 多线条激光线
        num_lines = np.random.randint(2, 5)  # 2-4条线
        
        for line in range(num_lines):
            y_base = int(height * (line + 1) / (num_lines + 1))
            amplitude = int(height * np.random.uniform(0.05, 0.15))
            frequency = np.random.uniform(1, 2) * 2 * np.pi / width
            phase = np.random.uniform(0, 2*np.pi)
            
            x = np.arange(0, width)
            y = y_base + amplitude * np.sin(frequency * x + phase)
            y = y.astype(int)
            
            for i in range(width):
                y_pos = y[i]
                if 0 <= y_pos < height:
                    for j in range(height):
                        dist = abs(j - y_pos)
                        intensity = line_intensity * np.exp(-(dist**2) / (2 * line_width**2))
                        image[j, i] += intensity
    
    elif line_type == LaserLineType.CROSS:
        # 交叉线条
        # 水平线
        y_h = height // 2
        for i in range(width):
            for j in range(height):
                dist = abs(j - y_h)
                intensity = line_intensity * np.exp(-(dist**2) / (2 * line_width**2))
                image[j, i] += intensity
        
        # 垂直线
        x_v = width // 2
        for j in range(height):
            for i in range(width):
                dist = abs(i - x_v)
                intensity = line_intensity * np.exp(-(dist**2) / (2 * line_width**2))
                image[j, i] += intensity * 0.8  # 稍微降低垂直线的强度，以便区分
    
    # 应用对比度调整
    if contrast != 1.0:
        mean_val = np.mean(image)
        image = mean_val + contrast * (image - mean_val)
    
    # 确保像素值在有效范围内
    image = np.clip(image, 0, 255)
    
    # 添加高斯噪声
    noise = np.random.normal(0, noise_level * 255, image.shape)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    # 保存图像
    if save_path:
        plt.figure(figsize=(10, 8))
        plt.imshow(image, cmap='gray')
        plt.title(f'模拟激光线图像 - {line_type.value}')
        plt.colorbar(label='强度')
        plt.savefig(save_path, dpi=100)
        plt.close()
        
        # 同时保存原始图像数据
        np.save(os.path.splitext(save_path)[0] + '.npy', image)
    
    return image


def generate_test_suite(output_dir='./test_images'):
    """
    生成一套完整的测试图像集
    
    参数:
        output_dir (str): 输出目录
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每种线型生成多个图像
    for line_type in LaserLineType:
        # 为每种线型生成3个不同参数的图像
        for i in range(3):
            # 随机参数
            noise_level = 0.05 + 0.15 * np.random.random()  # 0.05-0.2
            line_width = 3.0 + 5.0 * np.random.random()     # 3.0-8.0
            line_intensity = 150 + 100 * np.random.random()  # 150-250
            background_intensity = 30 + 40 * np.random.random()  # 30-70
            contrast = 0.8 + 0.4 * np.random.random()  # 0.8-1.2
            
            # 生成图像
            save_path = os.path.join(output_dir, f'laser_{line_type.value}_{i+1}.png')
            generate_advanced_laser_image(
                line_type=line_type,
                noise_level=noise_level,
                line_width=line_width,
                line_intensity=line_intensity,
                background_intensity=background_intensity,
                contrast=contrast,
                save_path=save_path
            )
            
            print(f'已生成图像: {save_path}')


def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='生成激光线测试图像')
    parser.add_argument('--type', type=str, choices=[t.value for t in LaserLineType], 
                        default='sine', help='激光线类型')
    parser.add_argument('--width', type=int, default=640, help='图像宽度')
    parser.add_argument('--height', type=int, default=480, help='图像高度')
    parser.add_argument('--noise', type=float, default=0.1, help='噪声水平 (0-1)')
    parser.add_argument('--line-width', type=float, default=5.0, help='激光线宽度')
    parser.add_argument('--line-intensity', type=int, default=200, help='激光线强度 (0-255)')
    parser.add_argument('--bg-intensity', type=int, default=50, help='背景强度 (0-255)')
    parser.add_argument('--contrast', type=float, default=1.0, help='对比度调整')
    parser.add_argument('--output', type=str, default='laser_test.png', help='输出文件名')
    parser.add_argument('--generate-all', action='store_true', help='生成所有类型的测试图像')
    
    args = parser.parse_args()
    
    if args.generate_all:
        generate_test_suite()
    else:
        line_type = LaserLineType(args.type)
        generate_advanced_laser_image(
            width=args.width,
            height=args.height,
            line_type=line_type,
            noise_level=args.noise,
            line_width=args.line_width,
            line_intensity=args.line_intensity,
            background_intensity=args.bg_intensity,
            contrast=args.contrast,
            save_path=args.output
        )
        print(f'已生成图像: {args.output}')


if __name__ == '__main__':
    main()
