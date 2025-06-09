# 比较不同的激光线中心提取方法

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import os
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter

# 导入各种方法
from improved_gray_gravity_method import ImprovedGrayGravityMethod
from steger_method import StegerMethod
from multi_scale_anisotropic_gaussian import MultiScaleAnisotropicGaussian
from shepard_interpolation_method import ShepardInterpolationMethod
from improved_gaussian_fitting import ImprovedGaussianFitting

# 如果存在深度学习模型，则导入深度学习方法
if os.path.exists("deep_learning_laser_center_model.h5"):
    from deep_learning_laser_center import DeepLearningLaserCenter

def generate_test_image(image_size=(400, 300), line_width=5, noise_level=15):
    """
    生成用于测试的合成激光线图像
    
    参数:
        image_size: 图像大小 (宽度, 高度)
        line_width: 激光线宽度
        noise_level: 噪声水平
        
    返回:
        image: 合成图像
        ground_truth: 真实中心点坐标列表
    """
    # 创建空白图像
    image = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    
    # 随机生成控制点
    num_points = 5
    x_points = np.linspace(50, image_size[0] - 50, num_points)
    y_points = np.random.randint(50, image_size[1] - 50, size=num_points)
    points = np.column_stack([x_points, y_points])
    
    # 使用样条插值生成平滑曲线
    tck, u = splprep([points[:, 0], points[:, 1]], s=0, k=min(3, num_points-1))
    x_new, y_new = splev(np.linspace(0, 1, 500), tck)
    
    # 将曲线点转换为整数坐标
    curve_points = np.column_stack([x_new, y_new])
    ground_truth = [(x, y) for x, y in curve_points]
    
    # 在图像上绘制激光线（高斯分布）
    for x, y in curve_points:
        x_int, y_int = int(x), int(y)
        if 0 <= x_int < image_size[0] and 0 <= y_int < image_size[1]:
            # 确定局部区域
            y_min = max(0, int(y - 3 * line_width))
            y_max = min(image_size[1], int(y + 3 * line_width))
            x_min = max(0, int(x - 3 * line_width))
            x_max = min(image_size[0], int(x + 3 * line_width))
            
            # 在局部区域内添加高斯分布
            for ly in range(y_min, y_max):
                for lx in range(x_min, x_max):
                    # 计算到中心点的距离
                    dist = np.sqrt((lx - x) ** 2 + (ly - y) ** 2)
                    # 使用高斯函数计算强度
                    intensity = 255 * np.exp(-0.5 * (dist / (line_width / 2)) ** 2)
                    # 更新像素值（取最大值）
                    image[ly, lx] = max(image[ly, lx], int(intensity))
    
    # 添加高斯噪声
    noise = np.random.normal(0, noise_level, image.shape).astype(np.int32)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    return image, ground_truth

def calculate_accuracy(extracted_centers, ground_truth, distance_threshold=2.0):
    """
    计算提取中心点的准确率和平均误差
    
    参数:
        extracted_centers: 提取的中心点列表
        ground_truth: 真实中心点列表
        distance_threshold: 距离阈值，小于此值视为正确匹配
        
    返回:
        accuracy: 准确率（0-100%）
        mean_error: 平均误差（像素）
    """
    if not extracted_centers or not ground_truth:
        return 0.0, float('inf')
    
    # 对提取的中心点和真实中心点进行排序（按x坐标）
    extracted_centers = sorted(extracted_centers, key=lambda p: p[0])
    ground_truth = sorted(ground_truth, key=lambda p: p[0])
    
    # 对真实中心点进行采样，使数量与提取的中心点相近
    if len(ground_truth) > len(extracted_centers):
        indices = np.linspace(0, len(ground_truth) - 1, len(extracted_centers)).astype(int)
        sampled_ground_truth = [ground_truth[i] for i in indices]
    else:
        sampled_ground_truth = ground_truth
    
    # 计算每个提取中心点到最近真实中心点的距离
    correct_matches = 0
    total_error = 0.0
    
    for ex, ey in extracted_centers:
        # 找到最近的真实中心点
        min_distance = float('inf')
        for gx, gy in sampled_ground_truth:
            distance = np.sqrt((ex - gx) ** 2 + (ey - gy) ** 2)
            if distance < min_distance:
                min_distance = distance
        
        # 如果距离小于阈值，视为正确匹配
        if min_distance < distance_threshold:
            correct_matches += 1
            total_error += min_distance
    
    # 计算准确率和平均误差
    accuracy = 100.0 * correct_matches / len(extracted_centers) if extracted_centers else 0.0
    mean_error = total_error / correct_matches if correct_matches > 0 else float('inf')
    
    return accuracy, mean_error

def compare_methods(test_image, ground_truth):
    """
    比较不同方法的性能
    
    参数:
        test_image: 测试图像
        ground_truth: 真实中心点坐标列表
        
    返回:
        results: 包含各方法结果的字典
    """
    results = {}
    
    # 1. 改进的灰度重心法
    print("Testing Improved Gray Gravity Method...")
    start_time = time.time()
    iggm = ImprovedGrayGravityMethod(window_size=7, threshold=30)
    iggm_centers = iggm.extract_centers(test_image)
    iggm_time = time.time() - start_time
    iggm_accuracy, iggm_error = calculate_accuracy(iggm_centers, ground_truth)
    results["Improved Gray Gravity"] = {
        "centers": iggm_centers,
        "time": iggm_time,
        "accuracy": iggm_accuracy,
        "error": iggm_error
    }
    
    # 2. Steger方法
    print("Testing Steger Method...")
    start_time = time.time()
    steger = StegerMethod(sigma=1.5, threshold=0.5)
    steger_centers = steger.extract_centers(test_image)
    steger_time = time.time() - start_time
    steger_accuracy, steger_error = calculate_accuracy(steger_centers, ground_truth)
    results["Steger"] = {
        "centers": steger_centers,
        "time": steger_time,
        "accuracy": steger_accuracy,
        "error": steger_error
    }
    
    # 3. 多尺度各向异性高斯方法
    print("Testing Multi-Scale Anisotropic Gaussian Method...")
    start_time = time.time()
    msag = MultiScaleAnisotropicGaussian(sigma_range=(1.0, 3.0), num_scales=3, threshold=30)
    msag_centers = msag.extract_centers(test_image)
    msag_time = time.time() - start_time
    msag_accuracy, msag_error = calculate_accuracy(msag_centers, ground_truth)
    results["Multi-Scale Anisotropic Gaussian"] = {
        "centers": msag_centers,
        "time": msag_time,
        "accuracy": msag_accuracy,
        "error": msag_error
    }
    
    # 4. Shepard插值方法
    print("Testing Shepard Interpolation Method...")
    start_time = time.time()
    shepard = ShepardInterpolationMethod(window_size=5, power_parameter=2.0, threshold=30)
    shepard_centers = shepard.extract_centers(test_image)
    shepard_time = time.time() - start_time
    shepard_accuracy, shepard_error = calculate_accuracy(shepard_centers, ground_truth)
    results["Shepard Interpolation"] = {
        "centers": shepard_centers,
        "time": shepard_time,
        "accuracy": shepard_accuracy,
        "error": shepard_error
    }
    
    # 5. 改进的高斯拟合方法
    print("Testing Improved Gaussian Fitting Method...")
    start_time = time.time()
    igf = ImprovedGaussianFitting(window_size=7, threshold=30, max_iterations=5)
    igf_centers = igf.extract_centers(test_image)
    igf_time = time.time() - start_time
    igf_accuracy, igf_error = calculate_accuracy(igf_centers, ground_truth)
    results["Improved Gaussian Fitting"] = {
        "centers": igf_centers,
        "time": igf_time,
        "accuracy": igf_accuracy,
        "error": igf_error
    }
    
    # 6. 深度学习方法（如果模型存在）
    if os.path.exists("deep_learning_laser_center_model.h5"):
        print("Testing Deep Learning Method...")
        start_time = time.time()
        dl_model = DeepLearningLaserCenter(model_path="deep_learning_laser_center_model.h5")
        dl_centers, _ = dl_model.predict(test_image)
        dl_time = time.time() - start_time
        dl_accuracy, dl_error = calculate_accuracy(dl_centers, ground_truth)
        results["Deep Learning"] = {
            "centers": dl_centers,
            "time": dl_time,
            "accuracy": dl_accuracy,
            "error": dl_error
        }
    
    return results

def visualize_comparison(test_image, results, ground_truth=None):
    """
    可视化比较结果
    
    参数:
        test_image: 测试图像
        results: 比较结果字典
        ground_truth: 真实中心点坐标列表（可选）
    """
    num_methods = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # 显示原始图像和真实中心点
    if ground_truth:
        vis_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
        for x, y in ground_truth:
            if 0 <= int(x) < test_image.shape[1] and 0 <= int(y) < test_image.shape[0]:
                cv2.circle(vis_image, (int(x), int(y)), 1, (0, 255, 0), -1)
        axes[0].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Ground Truth")
        axes[0].axis('off')
    else:
        axes[0].imshow(test_image, cmap='gray')
        axes[0].set_title("Test Image")
        axes[0].axis('off')
    
    # 显示各方法结果
    for i, (method_name, result) in enumerate(results.items(), 1):
        if i < len(axes):
            vis_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
            for x, y in result["centers"]:
                cv2.circle(vis_image, (int(x), int(y)), 1, (0, 0, 255), -1)
            
            axes[i].imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"{method_name}\nAcc: {result['accuracy']:.1f}%, Err: {result['error']:.2f}px\nTime: {result['time']:.3f}s")
            axes[i].axis('off')
    
    # 隐藏多余的子图
    for i in range(num_methods + 1, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_performance_comparison(results):
    """
    绘制性能比较图表
    
    参数:
        results: 比较结果字典
    """
    method_names = list(results.keys())
    accuracies = [results[name]["accuracy"] for name in method_names]
    errors = [results[name]["error"] for name in method_names]
    times = [results[name]["time"] for name in method_names]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 准确率柱状图
    axes[0].bar(method_names, accuracies, color='skyblue')
    axes[0].set_title('Accuracy Comparison')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha='center')
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 误差柱状图
    axes[1].bar(method_names, errors, color='salmon')
    axes[1].set_title('Mean Error Comparison')
    axes[1].set_ylabel('Mean Error (pixels)')
    for i, v in enumerate(errors):
        axes[1].text(i, v + 0.05, f"{v:.2f}px", ha='center')
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 处理时间柱状图
    axes[2].bar(method_names, times, color='lightgreen')
    axes[2].set_title('Processing Time Comparison')
    axes[2].set_ylabel('Time (seconds)')
    for i, v in enumerate(times):
        axes[2].text(i, v + 0.05, f"{v:.3f}s", ha='center')
    plt.setp(axes[2].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数
    """
    # 生成测试图像
    print("Generating test image...")
    test_image, ground_truth = generate_test_image(line_width=5, noise_level=15)
    
    # 比较不同方法
    print("Comparing methods...")
    results = compare_methods(test_image, ground_truth)
    
    # 可视化比较结果
    print("Visualizing comparison...")
    visualize_comparison(test_image, results, ground_truth)
    
    # 绘制性能比较图表
    print("Plotting performance comparison...")
    plot_performance_comparison(results)
    
    # 打印详细结果
    print("\nDetailed Results:")
    for method_name, result in results.items():
        print(f"\n{method_name}:")
        print(f"  - Number of centers: {len(result['centers'])}")
        print(f"  - Accuracy: {result['accuracy']:.2f}%")
        print(f"  - Mean Error: {result['error']:.4f} pixels")
        print(f"  - Processing Time: {result['time']:.4f} seconds")

if __name__ == "__main__":
    main()
