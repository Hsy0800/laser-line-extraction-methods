# 激光线中心提取方法集合

本项目实现了多种创新性的激光线中心提取方法，用于高精度亚像素级别的激光线中心提取。这些方法可应用于结构光三维测量、激光焊接跟踪、机器视觉检测等领域。

## 项目结构

```
├── improved_gray_gravity_method.py  # 改进的灰度重心法
├── steger_method.py                # Steger方法
├── multi_scale_anisotropic_gaussian.py  # 多尺度各向异性高斯核方法
├── deep_learning_laser_center.py   # 基于深度学习的方法
├── shepard_interpolation_method.py # 基于Shepard插值的方法
├── improved_gaussian_fitting.py    # 改进的高斯曲线拟合方法
├── compare_methods.py              # 方法比较和评估
├── examples/                       # 示例脚本目录
│   ├── basic_usage.py              # 基本使用示例
│   ├── performance_comparison.py   # 性能对比示例
│   └── real_world_application.py   # 实际应用示例
├── requirements.txt                # 项目依赖
├── LICENSE                         # MIT许可证
└── README.md                       # 项目说明文档
```

## 实现的方法

### 1. 改进的灰度重心法 (IGGM)

**特点：**
- 考虑矩形区域而非单列像素，提高抗噪性
- 使用自适应窗口大小，适应不同宽度的激光线
- 计算速度快，适合实时应用
- 精度适中，一般可达到0.1像素级别

### 2. Steger方法

**特点：**
- 基于二阶偏导数和Hessian矩阵分析
- 高精度，可达到0.05像素级别
- 对噪声具有较好的鲁棒性
- 计算复杂度较高，速度较慢
- 对参数选择较为敏感

### 3. 多尺度各向异性高斯核方法

**特点：**
- 使用各向异性高斯核对激光线进行建模
- 通过多尺度分析提高精度和鲁棒性
- 在高噪声环境下仍能保持高精度
- 计算复杂度中等，精度高

### 4. 基于深度学习的方法

**特点：**
- 使用U-Net架构的深度神经网络直接从图像中提取激光线中心
- 能够处理复杂环境下的激光线图像，如低对比度、噪声干扰等
- 通过端到端的方式学习最优的中心提取策略
- 可以适应不同的激光线形态和环境条件
- 需要训练数据和GPU加速

### 5. 基于Shepard插值的方法

**特点：**
- 使用Shepard插值实现亚像素精度的中心点提取
- 结合灰度重心法和插值技术，提高精度
- 对噪声和不均匀光照具有较好的鲁棒性
- 计算效率高，适合实时应用

### 6. 改进的高斯曲线拟合方法

**特点：**
- 使用改进的高斯模型对激光线剖面进行拟合
- 考虑背景光照和噪声影响
- 通过迭代优化提高拟合精度
- 可以处理不同宽度和强度的激光线
- 精度高，可达到0.01像素级别

## 使用方法

每个方法都被实现为独立的Python类，可以单独使用或通过比较工具进行对比。

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/Hsy0800/laser-line-extraction-methods.git
cd laser-line-extraction-methods

# 安装依赖
pip install -r requirements.txt
```

### 基本使用示例

```python
# 以改进的灰度重心法为例
from improved_gray_gravity_method import ImprovedGrayGravityMethod
import cv2

# 加载图像
image = cv2.imread('laser_image.png', cv2.IMREAD_GRAYSCALE)

# 创建提取器实例
extractor = ImprovedGrayGravityMethod(window_size=7, threshold=30)

# 提取中心点
centers = extractor.extract_centers(image)

# 可视化结果
extractor.visualize_centers(image, centers)
```

### 示例脚本

项目提供了多个示例脚本，展示如何使用这些方法：

#### 1. 基本使用示例

```bash
python examples/basic_usage.py
```

这个脚本展示了如何使用各种方法提取激光线中心点，并可视化结果。

#### 2. 性能对比示例

```bash
python examples/performance_comparison.py
```

这个脚本对比了不同方法的性能，包括准确率、误差和处理时间，并生成对比图表。

#### 3. 实际应用示例

```bash
# 处理单张图像
python examples/real_world_application.py --input path/to/image.jpg --method iggm --output result.jpg

# 处理视频
python examples/real_world_application.py --input path/to/video.mp4 --method steger --output processed_video.avi --video

# 批量处理图像
python examples/real_world_application.py --input path/to/images_folder --method gaussian --output results_folder --batch
```

这个脚本展示了如何在实际项目中应用这些方法，支持处理单张图像、视频和批量图像。

### 方法比较

使用`compare_methods.py`可以对不同方法进行比较和评估：

```bash
python compare_methods.py
```

## 性能比较

不同方法在精度、速度和鲁棒性方面各有优劣：

| 方法 | 精度 | 速度 | 抗噪性 | 适用场景 |
|------|------|------|--------|----------|
| 改进的灰度重心法 | 中 | 快 | 中 | 实时应用，对精度要求不是特别高 |
| Steger方法 | 高 | 慢 | 高 | 高精度测量，对速度要求不高 |
| 多尺度各向异性高斯核 | 高 | 中 | 高 | 噪声环境下的高精度测量 |
| 深度学习方法 | 高 | 快(GPU) | 高 | 复杂环境，有大量训练数据 |
| Shepard插值方法 | 中高 | 快 | 中高 | 实时应用中需要较高精度 |
| 改进的高斯曲线拟合 | 高 | 中 | 高 | 高精度测量，激光线形状复杂 |

## 方法选择建议

根据不同的应用场景，可以选择不同的方法：

1. **实时应用**（如激光焊接跟踪）：
   - 改进的灰度重心法
   - Shepard插值方法

2. **高精度测量**（如结构光三维重建）：
   - Steger方法
   - 改进的高斯曲线拟合方法

3. **噪声环境**（如工业现场）：
   - 多尺度各向异性高斯核方法
   - 改进的高斯曲线拟合方法

4. **复杂环境**（如不均匀光照、复杂背景）：
   - 深度学习方法（如有训练数据）
   - 多尺度各向异性高斯核方法

## 开发指南

### 环境要求

- Python 3.6+
- OpenCV 4.0+
- NumPy
- SciPy
- Matplotlib
- TensorFlow 2.0+ (仅用于深度学习方法)

### 扩展新方法

如果要添加新的激光线中心提取方法，请遵循以下步骤：

1. 创建新的Python文件，实现一个包含`extract_centers`和`visualize_centers`方法的类
2. 在`compare_methods.py`中添加新方法的评估代码
3. 更新README.md，添加新方法的说明

## 贡献

欢迎贡献新的方法或改进现有方法！请遵循以下步骤：

1. Fork 本仓库
2. 创建新的分支 (`git checkout -b feature/your-feature`)
3. 提交更改 (`git commit -am 'Add some feature'`)
4. 推送到分支 (`git push origin feature/your-feature`)
5. 创建新的 Pull Request

## 许可证

MIT License