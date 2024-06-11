
import cv2
import numpy as np

def preprocess_image(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 调整图像大小
    image = cv2.resize(image, (128, 128))
    # 归一化
    image = image / 255.0
    # 增加维度，以适应模型输入
    image = np.expand_dims(image, axis=0)  # 形状变为 (1, 128, 128)
    image = image.astype(np.float32)
    return image