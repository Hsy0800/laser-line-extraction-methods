# 激光线中心提取方法实际应用示例
# 本示例展示如何在实际项目中使用激光线中心提取方法

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

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

# 创建提取器工厂
def create_extractor(method_name, **kwargs):
    """根据方法名称创建相应的提取器实例"""
    extractors = {
        'iggm': ImprovedGrayGravityMethod,
        'steger': StegerMethod,
        'msag': MultiScaleAnisotropicGaussian,
        'shepard': ShepardInterpolationMethod,
        'gaussian': ImprovedGaussianFitting
    }
    
    if method_name.lower() == 'dl' and has_deep_learning:
        return DeepLearningLaserCenter(**kwargs)
    
    if method_name.lower() in extractors:
        return extractors[method_name.lower()](**kwargs)
    else:
        raise ValueError(f"未知的方法名称: {method_name}，可用方法: {list(extractors.keys())}")

# 处理单张图像
def process_single_image(image_path, method_name, params=None, visualize=True, save_path=None):
    """处理单张图像并提取激光线中心"""
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 创建提取器
    params = params or {}
    extractor = create_extractor(method_name, **params)
    
    # 提取中心点
    print(f"使用 {method_name} 方法提取中心点...")
    centers = extractor.extract_centers(image)
    print(f"提取到 {len(centers)} 个中心点")
    
    # 可视化结果
    if visualize:
        plt.figure(figsize=(10, 8))
        
        # 显示原始图像
        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title('原始激光线图像')
        
        # 显示提取结果
        plt.subplot(122)
        result_image = extractor.visualize_centers(image, centers, return_image=True)
        plt.imshow(result_image)
        plt.title(f'{method_name} 提取结果')
        
        plt.tight_layout()
        
        # 保存结果
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"结果已保存至: {save_path}")
        
        plt.show()
    
    return centers

# 处理视频
def process_video(video_path, method_name, params=None, output_path=None, frame_step=1):
    """处理视频并提取每一帧的激光线中心"""
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建提取器
    params = params or {}
    extractor = create_extractor(method_name, **params)
    
    # 创建输出视频
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 处理每一帧
    frame_count = 0
    all_centers = []
    
    print(f"开始处理视频，共 {total_frames} 帧...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 每隔frame_step帧处理一次
        if frame_count % frame_step == 0:
            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 提取中心点
            centers = extractor.extract_centers(gray)
            all_centers.append(centers)
            
            # 可视化结果
            if output_path:
                # 在彩色帧上绘制中心点
                for x, y in centers:
                    cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)
                
                # 添加帧信息
                cv2.putText(frame, f"Frame: {frame_count}, Centers: {len(centers)}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 写入输出视频
                out.write(frame)
            
            # 显示进度
            if frame_count % (frame_step * 10) == 0:
                print(f"已处理 {frame_count}/{total_frames} 帧 ({frame_count/total_frames*100:.1f}%)")
        
        frame_count += 1
    
    # 释放资源
    cap.release()
    if output_path:
        out.release()
        print(f"处理后的视频已保存至: {output_path}")
    
    print(f"视频处理完成，共处理 {len(all_centers)} 帧")
    return all_centers

# 批量处理图像
def batch_process_images(image_dir, method_name, params=None, output_dir=None, extensions=('.png', '.jpg', '.jpeg')):
    """批量处理目录中的所有图像"""
    # 检查目录是否存在
    if not os.path.isdir(image_dir):
        raise ValueError(f"图像目录不存在: {image_dir}")
    
    # 创建输出目录
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有图像文件
    image_files = []
    for ext in extensions:
        image_files.extend([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(ext)
        ])
    
    if not image_files:
        print(f"在目录 {image_dir} 中未找到图像文件")
        return
    
    # 创建提取器
    params = params or {}
    extractor = create_extractor(method_name, **params)
    
    # 处理每张图像
    results = {}
    for i, image_path in enumerate(image_files):
        print(f"处理图像 {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"警告: 无法读取图像 {image_path}，跳过")
            continue
        
        # 提取中心点
        centers = extractor.extract_centers(image)
        results[image_path] = centers
        
        # 保存结果
        if output_dir:
            # 生成输出文件名
            base_name = os.path.basename(image_path)
            name, ext = os.path.splitext(base_name)
            output_path = os.path.join(output_dir, f"{name}_{method_name}{ext}")
            
            # 可视化并保存
            result_image = extractor.visualize_centers(image, centers, return_image=True)
            cv2.imwrite(output_path, result_image)
    
    print(f"批处理完成，共处理 {len(results)} 张图像")
    return results

# 主函数
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='激光线中心提取方法实际应用示例')
    parser.add_argument('--input', '-i', required=True, help='输入图像或视频文件路径，或图像目录')
    parser.add_argument('--method', '-m', default='iggm', 
                        choices=['iggm', 'steger', 'msag', 'shepard', 'gaussian', 'dl'],
                        help='使用的提取方法')
    parser.add_argument('--output', '-o', help='输出文件或目录路径')
    parser.add_argument('--batch', '-b', action='store_true', help='批处理模式（输入为目录）')
    parser.add_argument('--video', '-v', action='store_true', help='视频处理模式')
    parser.add_argument('--frame-step', type=int, default=1, help='视频处理时的帧间隔')
    parser.add_argument('--window-size', type=int, default=7, help='窗口大小参数')
    parser.add_argument('--threshold', type=int, default=30, help='阈值参数')
    parser.add_argument('--sigma', type=float, default=1.5, help='高斯sigma参数（用于Steger方法）')
    
    args = parser.parse_args()
    
    # 设置方法参数
    method_params = {
        'window_size': args.window_size,
        'threshold': args.threshold
    }
    
    if args.method == 'steger':
        method_params['sigma'] = args.sigma
    elif args.method == 'msag':
        method_params['scales'] = [1.0, 1.5, 2.0]
    elif args.method == 'dl':
        # 检查是否有预训练模型
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 'models/laser_center_model.h5')
        if os.path.exists(model_path):
            method_params['model_path'] = model_path
    
    # 根据模式选择处理方法
    if args.batch:
        # 批处理模式
        print(f"使用 {args.method} 方法批量处理目录 {args.input} 中的图像...")
        batch_process_images(args.input, args.method, method_params, args.output)
    elif args.video:
        # 视频处理模式
        print(f"使用 {args.method} 方法处理视频 {args.input}...")
        process_video(args.input, args.method, method_params, args.output, args.frame_step)
    else:
        # 单图像处理模式
        print(f"使用 {args.method} 方法处理图像 {args.input}...")
        process_single_image(args.input, args.method, method_params, True, args.output)

if __name__ == "__main__":
    main()