# 测试图像

此目录用于存放测试激光线图像。

## 使用说明

1. 您可以将自己的激光线图像放在此目录中进行测试
2. 支持的图像格式：PNG, JPG, JPEG, BMP
3. 推荐使用灰度图像，如果使用彩色图像，程序会自动转换为灰度

## 示例图像生成

如果没有实际的激光线图像，可以使用以下脚本生成模拟的激光线图像：

```python
# 在examples目录下运行
python basic_usage.py
# 或
python performance_comparison.py
```

这些脚本会生成模拟的激光线图像并保存在当前目录。