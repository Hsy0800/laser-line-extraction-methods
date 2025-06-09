# 激光线提取方法总结与实现示例

本仓库包含了多种亚像素级激光线中心提取方法的Python实现，以及它们的性能比较。这些方法广泛应用于结构光三维重建、激光焊接跟踪、工业检测等领域。

## 实现的方法

本仓库实现了以下几种激光线中心提取方法：

1. **改进的灰度重心法 (Improved Gray Gravity Method)**
   - 基于灰度加权平均的简单高效方法
   - 适用于噪声较小的场景
   - 实现文件：`improved_gray_gravity_method.py`

2. **Steger方法 (Steger Method)**
   - 基于二阶导数矩阵特征值和特征向量的方法
   - 高精度，适用于各种宽度的激光线
   - 实现文件：`steger_method.py`

3. **多尺度各向异性高斯方法 (Multi-Scale Anisotropic Gaussian)**
   - 使用多个尺度的各向异性高斯滤波器
   - 适应不同宽度和方向的激光线
   - 实现文件：`multi_scale_anisotropic_gaussian.py`

4. **Shepard插值方法 (Shepard Interpolation Method)**
   - 使用Shepard插值进行亚像素精度提取
   - 考虑局部邻域内的灰度分布
   - 实现文件：`shepard_interpolation_method.py`

5. **改进的高斯拟合方法 (Improved Gaussian Fitting)**
   - 使用标准高斯模型和非对称高斯模型进行拟合
   - 迭代优化提高亚像素精度
   - 实现文件：`improved_gaussian_fitting.py`

6. **深度学习方法 (Deep Learning Method)**
   - 使用U-Net架构进行像素级分割
   - 端到端的中心线提取
   - 实现文件：`deep_learning_laser_center.py`

## 方法比较

各方法的性能比较包括以下几个方面：

- **准确率**：提取中心点与真实中心点的匹配程度
- **平均误差**：提取中心点与真实中心点的平均距离
- **处理时间**：算法执行所需的时间

比较代码实现在 `compare_methods.py` 文件中。

## 使用方法

### 环境要求

```
python >= 3.6
numpy
opencv-python
scipy
matplotlib
tensorflow >= 2.0 (仅深度学习方法需要)
```

### 单个方法使用示例

每个方法文件都包含了一个 `example_usage()` 函数，可以直接运行查看效果：

```python
# 以改进的高斯拟合方法为例
from improved_gaussian_fitting import ImprovedGaussianFitting, example_usage

# 运行示例
image, centers = example_usage()

# 或者自定义参数使用
gaussian_fitting = ImprovedGaussianFitting(window_size=7, threshold=30)
centers = gaussian_fitting.extract_centers(your_image)
gaussian_fitting.visualize_centers(your_image, centers)
```

### 比较不同方法

```python
from compare_methods import main

# 运行比较
main()
```

## 方法选择建议

- **计算效率优先**：改进的灰度重心法
- **精度优先**：Steger方法、改进的高斯拟合方法
- **复杂场景**：多尺度各向异性高斯方法、深度学习方法
- **平衡精度和效率**：Shepard插值方法

## 参考文献

1. Steger, C. (1998). An unbiased detector of curvilinear structures. IEEE Transactions on Pattern Analysis and Machine Intelligence, 20(2), 113-125.
2. Forest, J., & Salvi, J. (2002). A review of laser scanning three-dimensional digitizers. In Proceedings of IEEE/RSJ International Conference on Intelligent Robots and Systems.
3. Usamentiaga, R., Molleda, J., & Garcia, D. F. (2012). Fast and robust laser stripe extraction for 3D reconstruction in industrial environments. Machine Vision and Applications, 23(1), 179-196.
4. Haug, K., & Pritschow, G. (1998). Robust laser-stripe sensor for automated weld-seam-tracking in the shipbuilding industry. In Proceedings of the 24th Annual Conference of the IEEE Industrial Electronics Society.

## 贡献

欢迎提交问题和改进建议！

## 许可证

MIT
