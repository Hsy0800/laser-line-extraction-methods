# 深度学习方法用于亚像素激光线中心提取
# 参考: 基于深度学习的激光条纹中心提取方法

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter
import os

class DeepLearningLaserCenter:
    """
    深度学习方法用于亚像素激光线中心提取
    
    特点:
    1. 使用U-Net架构进行像素级分割
    2. 端到端的中心线提取
    3. 对复杂背景和噪声具有较强的鲁棒性
    4. 可以处理不同宽度和强度的激光线
    """
    
    def __init__(self, model_path=None, input_shape=(256, 256, 1)):
        """
        初始化深度学习激光中心提取方法
        
        参数:
            model_path: 预训练模型路径，如果为None则创建新模型
            input_shape: 输入图像形状
        """
        self.input_shape = input_shape
        
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            print(f"Loaded model from {model_path}")
        else:
            self.model = self._build_unet_model()
            print("Created new model")
    
    def _build_unet_model(self):
        """
        构建U-Net模型
        
        返回:
            model: 编译好的U-Net模型
        """
        # 输入层
        inputs = Input(self.input_shape)
        
        # 编码器部分
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        
        # 瓶颈部分
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        
        # 解码器部分
        up5 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv3], axis=-1)
        conv5 = Conv2D(256, 3, activation='relu', padding='same')(up5)
        conv5 = Conv2D(256, 3, activation='relu', padding='same')(conv5)
        
        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv2], axis=-1)
        conv6 = Conv2D(128, 3, activation='relu', padding='same')(up6)
        conv6 = Conv2D(128, 3, activation='relu', padding='same')(conv6)
        
        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=-1)
        conv7 = Conv2D(64, 3, activation='relu', padding='same')(up7)
        conv7 = Conv2D(64, 3, activation='relu', padding='same')(conv7)
        
        # 输出层 - 单通道二值掩码
        outputs = Conv2D(1, 1, activation='sigmoid')(conv7)
        
        # 构建模型
        model = Model(inputs=inputs, outputs=outputs)
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_images, train_masks, validation_split=0.2, epochs=50, batch_size=8):
        """
        训练模型
        
        参数:
            train_images: 训练图像数组，形状为(n_samples, height, width, channels)
            train_masks: 训练掩码数组，形状为(n_samples, height, width, 1)
            validation_split: 验证集比例
            epochs: 训练轮数
            batch_size: 批次大小
            
        返回:
            history: 训练历史
        """
        # 确保输入数据形状正确
        if len(train_images.shape) == 3:
            train_images = np.expand_dims(train_images, axis=-1)
        if len(train_masks.shape) == 3:
            train_masks = np.expand_dims(train_masks, axis=-1)
        
        # 归一化图像
        train_images = train_images.astype('float32') / 255.0
        
        # 训练模型
        history = self.model.fit(
            train_images,
            train_masks,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def save_model(self, model_path):
        """
        保存模型
        
        参数:
            model_path: 模型保存路径
        """
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def predict(self, image):
        """
        预测图像中的激光线中心
        
        参数:
            image: 输入图像
            
        返回:
            centers: 中心点坐标列表
            center_mask: 中心线掩码
        """
        # 预处理图像
        if len(image.shape) > 2:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
        
        # 调整图像大小为模型输入尺寸
        original_shape = gray_image.shape
        resized_image = cv2.resize(gray_image, (self.input_shape[1], self.input_shape[0]))
        
        # 归一化并添加批次维度
        input_image = resized_image.astype('float32') / 255.0
        input_image = np.expand_dims(input_image, axis=0)  # 添加批次维度
        input_image = np.expand_dims(input_image, axis=-1)  # 添加通道维度
        
        # 预测中心线掩码
        predicted_mask = self.model.predict(input_image)[0, :, :, 0]
        
        # 将掩码调整回原始尺寸
        center_mask = cv2.resize(predicted_mask, (original_shape[1], original_shape[0]))
        
        # 二值化掩码
        binary_mask = (center_mask > 0.5).astype(np.uint8) * 255
        
        # 细化为单像素宽度的线
        skeleton = cv2.ximgproc.thinning(binary_mask)
        
        # 提取中心点坐标
        y_indices, x_indices = np.where(skeleton > 0)
        centers = list(zip(x_indices, y_indices))
        
        return centers, center_mask
    
    def visualize_centers(self, image, centers, center_mask=None):
        """
        可视化提取的中心点
        
        参数:
            image: 原始图像
            centers: 中心点列表
            center_mask: 中心线掩码（可选）
        """
        if len(image.shape) > 2:
            vis_image = image.copy()
        else:
            vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # 绘制中心点
        for x, y in centers:
            cv2.circle(vis_image, (int(x), int(y)), 1, (0, 0, 255), -1)
        
        plt.figure(figsize=(15, 5))
        
        # 显示原始图像和中心点
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title('Extracted Laser Line Centers')
        plt.axis('off')
        
        # 显示原始图像
        plt.subplot(1, 3, 2)
        if len(image.shape) > 2:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # 显示预测的中心线掩码
        if center_mask is not None:
            plt.subplot(1, 3, 3)
            plt.imshow(center_mask, cmap='jet')
            plt.title('Predicted Center Mask')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def generate_synthetic_dataset(num_samples=100, image_size=(256, 256), line_width_range=(3, 8), noise_level_range=(5, 20)):
    """
    生成合成数据集用于训练深度学习模型
    
    参数:
        num_samples: 样本数量
        image_size: 图像大小
        line_width_range: 激光线宽度范围
        noise_level_range: 噪声水平范围
        
    返回:
        images: 合成图像数组
        masks: 中心线掩码数组
        ground_truth: 真实中心点坐标列表
    """
    images = []
    masks = []
    ground_truth = []
    
    for i in range(num_samples):
        # 创建空白图像和掩码
        image = np.zeros(image_size, dtype=np.uint8)
        mask = np.zeros(image_size, dtype=np.uint8)
        
        # 随机生成控制点
        num_points = np.random.randint(3, 8)
        x_points = np.linspace(20, image_size[1] - 20, num_points)
        y_points = np.random.randint(20, image_size[0] - 20, size=num_points)
        points = np.column_stack([x_points, y_points])
        
        # 使用样条插值生成平滑曲线
        tck, u = splprep([points[:, 0], points[:, 1]], s=0, k=min(3, num_points-1))
        x_new, y_new = splev(np.linspace(0, 1, 500), tck)
        
        # 将曲线点转换为整数坐标
        curve_points = np.column_stack([x_new, y_new]).astype(np.int32)
        curve_points = np.array([p for p in curve_points if 0 <= p[0] < image_size[1] and 0 <= p[1] < image_size[0]])
        
        # 保存真实中心点坐标
        gt_centers = [(x, y) for x, y in curve_points]
        ground_truth.append(gt_centers)
        
        # 在掩码上绘制中心线
        for x, y in curve_points:
            mask[y, x] = 255
        
        # 随机选择激光线宽度
        line_width = np.random.uniform(line_width_range[0], line_width_range[1])
        
        # 在图像上绘制激光线（高斯分布）
        for x, y in curve_points:
            # 确定局部区域
            y_min = max(0, int(y - 3 * line_width))
            y_max = min(image_size[0], int(y + 3 * line_width))
            x_min = max(0, int(x - 3 * line_width))
            x_max = min(image_size[1], int(x + 3 * line_width))
            
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
        noise_level = np.random.uniform(noise_level_range[0], noise_level_range[1])
        noise = np.random.normal(0, noise_level, image.shape).astype(np.int32)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # 将图像和掩码添加到数据集
        images.append(image)
        masks.append(mask)
    
    return np.array(images), np.array(masks), ground_truth

def example_usage():
    """
    示例用法
    """
    # 创建模型
    model_path = "deep_learning_laser_center_model.h5"
    dl_model = DeepLearningLaserCenter(model_path=None)  # 不加载预训练模型
    
    # 生成合成数据集
    print("Generating synthetic dataset...")
    images, masks, ground_truth = generate_synthetic_dataset(num_samples=100)
    
    # 训练模型
    print("Training model...")
    history = dl_model.train(images, masks, epochs=20, batch_size=8)
    
    # 保存模型
    dl_model.save_model(model_path)
    
    # 在测试图像上进行预测
    print("Predicting on test image...")
    test_image = images[-1]  # 使用最后一个图像作为测试
    centers, center_mask = dl_model.predict(test_image)
    
    # 可视化结果
    dl_model.visualize_centers(test_image, centers, center_mask)
    
    return dl_model, test_image, centers, center_mask

if __name__ == "__main__":
    # 检查是否存在预训练模型
    model_path = "deep_learning_laser_center_model.h5"
    if os.path.exists(model_path):
        # 加载预训练模型并进行预测
        dl_model = DeepLearningLaserCenter(model_path=model_path)
        
        # 创建一个测试图像
        image = np.zeros((256, 256), dtype=np.uint8)
        for x in range(50, 200):
            y = int(100 + 50 * np.sin((x - 50) / 50))
            for dy in range(-5, 6):
                intensity = 255 * np.exp(-0.5 * (dy ** 2) / 4)
                if 0 <= y + dy < image.shape[0]:
                    image[y + dy, x] = min(255, int(intensity))
        
        # 添加噪声
        noise = np.random.normal(0, 15, image.shape).astype(np.int32)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # 预测中心点
        centers, center_mask = dl_model.predict(image)
        
        # 可视化结果
        dl_model.visualize_centers(image, centers, center_mask)
    else:
        # 运行示例训练和预测
        example_usage()
