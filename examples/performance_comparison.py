# 激光线中心提取方法性能对比示例

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
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
    centers = [(int(xi), float(yi)) for xi, yi in zip(x, y) if 0 <= xi < width and 0 <= yi < height]
    return image, centers

# 计算提取中心点的准确性
def calculate_accuracy(extracted_centers, ground_truth, max_distance=2.0):
    if not extracted_centers or not ground_truth:
        return 0.0, float('inf')
    
    # 将提取的中心点和真实中心点转换为numpy数组
    extracted = np.array(extracted_centers)
    truth = np.array(ground_truth)
    
    # 计算每个提取点到最近真实点的距离
    min_distances = []
    matched_count = 0
    
    for i, (x, y) in enumerate(extracted):
        # 计算当前点到所有真实点的距离
        distances = np.sqrt((truth[:, 0] - x) ** 2 + (truth[:, 1] - y) ** 2)
        min_dist = np.min(distances)
        min_distances.append(min_dist)
        
        # 如果距离小于阈值，认为匹配成功
        if min_dist <= max_distance:
            matched_count += 1
    
    # 计算准确率和平均误差
    accuracy = matched_count / len(extracted) if extracted.shape[0] > 0 else 0
    mean_error = np.mean(min_distances) if min_distances else float('inf')
    
    return accuracy, mean_error

# 比较不同方法的性能
def compare_methods(image, ground_truth):
    results = {}
    
    # 改进的灰度重心法
    print("测试改进的灰度重心法...")
    iggm = ImprovedGrayGravityMethod(window_size=7, threshold=30)
    start_time = time.time()
    iggm_centers = iggm.extract_centers(image)
    iggm_time = time.time() - start_time
    iggm_accuracy, iggm_error = calculate_accuracy(iggm_centers, ground_truth)
    results['IGGM'] = {
        'centers': iggm_centers,
        'time': iggm_time,
        'accuracy': iggm_accuracy,
        'error': iggm_error
    }
    
    # Steger方法
    print("测试Steger方法...")
    steger = StegerMethod(sigma=1.5, threshold=5.0)
    start_time = time.time()
    steger_centers = steger.extract_centers(image)
    steger_time = time.time() - start_time
    steger_accuracy, steger_error = calculate_accuracy(steger_centers, ground_truth)
    results['Steger'] = {
        'centers': steger_centers,
        'time': steger_time,
        'accuracy': steger_accuracy,
        'error': steger_error
    }
    
    # 多尺度各向异性高斯核方法
    print("测试多尺度各向异性高斯核方法...")
    msag = MultiScaleAnisotropicGaussian(scales=[1.0, 1.5, 2.0], threshold=30)
    start_time = time.time()
    msag_centers = msag.extract_centers(image)
    msag_time = time.time() - start_time
    msag_accuracy, msag_error = calculate_accuracy(msag_centers, ground_truth)
    results['MSAG'] = {
        'centers': msag_centers,
        'time': msag_time,
        'accuracy': msag_accuracy,
        'error': msag_error
    }
    
    # Shepard插值方法
    print("测试Shepard插值方法...")
    sim = ShepardInterpolationMethod(window_size=7, threshold=30)
    start_time = time.time()
    sim_centers = sim.extract_centers(image)
    sim_time = time.time() - start_time
    sim_accuracy, sim_error = calculate_accuracy(sim_centers, ground_truth)
    results['Shepard'] = {
        'centers': sim_centers,
        'time': sim_time,
        'accuracy': sim_accuracy,
        'error': sim_error
    }
    
    # 改进的高斯曲线拟合方法
    print("测试改进的高斯曲线拟合方法...")
    igf = ImprovedGaussianFitting(window_size=7, threshold=30)
    start_time = time.time()
    igf_centers = igf.extract_centers(image)
    igf_time = time.time() - start_time
    igf_accuracy, igf_error = calculate_accuracy(igf_centers, ground_truth)
    results['IGF'] = {
        'centers': igf_centers,
        'time': igf_time,
        'accuracy': igf_accuracy,
        'error': igf_error
    }
    
    # 深度学习方法（如果可用）
    if has_deep_learning:
        print("测试深度学习方法...")
        try:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     'models/laser_center_model.h5')
            if os.path.exists(model_path):
                dl = DeepLearningLaserCenter(model_path=model_path)
                start_time = time.time()
                dl_centers = dl.extract_centers(image)
                dl_time = time.time() - start_time
                dl_accuracy, dl_error = calculate_accuracy(dl_centers, ground_truth)
                results['DL'] = {
                    'centers': dl_centers,
                    'time': dl_time,
                    'accuracy': dl_accuracy,
                    'error': dl_error
                }
            else:
                print("未找到预训练模型，跳过深度学习方法测试。")
        except Exception as e:
            print(f"深度学习方法测试失败: {e}")
    
    return results

# 可视化比较结果
def visualize_comparison(image, results):
    methods = list(results.keys())
    n_methods = len(methods)
    
    # 创建图形
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 显示原始图像
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('原始激光线图像')
    axes[0].axis('off')
    
    # 显示各方法结果
    for i, method in enumerate(methods):
        if i >= 5:  # 最多显示5个方法
            break
            
        ax = axes[i+1]
        result = results[method]
        
        # 显示图像和提取的中心点
        ax.imshow(image, cmap='gray')
        centers = result['centers']
        x = [c[0] for c in centers]
        y = [c[1] for c in centers]
        ax.scatter(x, y, c='r', s=1)
        
        # 显示性能指标
        title = f"{method}\n准确率: {result['accuracy']:.2f}\n误差: {result['error']:.2f}像素\n时间: {result['time']:.3f}秒"
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('method_comparison_visualization.png', dpi=300)
    plt.show()

# 绘制性能对比图表
def plot_performance_comparison(results):
    methods = list(results.keys())
    
    # 提取性能指标
    accuracies = [results[m]['accuracy'] for m in methods]
    errors = [results[m]['error'] for m in methods]
    times = [results[m]['time'] for m in methods]
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 准确率对比
    axes[0].bar(methods, accuracies, color='skyblue')
    axes[0].set_title('准确率对比')
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel('准确率')
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    # 误差对比
    axes[1].bar(methods, errors, color='salmon')
    axes[1].set_title('平均误差对比')
    axes[1].set_ylabel('误差 (像素)')
    for i, v in enumerate(errors):
        axes[1].text(i, v + 0.02, f"{v:.2f}", ha='center')
    
    # 处理时间对比
    axes[2].bar(methods, times, color='lightgreen')
    axes[2].set_title('处理时间对比')
    axes[2].set_ylabel('时间 (秒)')
    for i, v in enumerate(times):
        axes[2].text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig('performance_comparison_charts.png', dpi=300)
    plt.show()

# 主函数
def main():
    # 生成测试图像
    print("生成测试激光线图像...")
    image, ground_truth = generate_test_image(noise_level=15)
    
    # 比较不同方法
    print("比较不同的激光线中心提取方法...")
    results = compare_methods(image, ground_truth)
    
    # 可视化比较结果
    print("可视化比较结果...")
    visualize_comparison(image, results)
    
    # 绘制性能对比图表
    print("绘制性能对比图表...")
    plot_performance_comparison(results)
    
    # 打印详细结果
    print("\n详细结果:")
    for method, result in results.items():
        print(f"\n{method}:")
        print(f"  提取中心点数量: {len(result['centers'])}")
        print(f"  准确率: {result['accuracy']:.4f}")
        print(f"  平均误差: {result['error']:.4f} 像素")
        print(f"  处理时间: {result['time']:.4f} 秒")
    
    print("\n所有测试完成，结果已保存为图像。")

if __name__ == "__main__":
    main()