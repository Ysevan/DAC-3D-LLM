import cv2
import numpy as np

def tenengrad(image):
    assert image is not None, "Image is empty"

    # 转换为灰度图
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image.copy()

    # 计算Sobel梯度
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅值
    G = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    # 计算均值（排除边缘像素）
    height, width = gray_img.shape
    return cv2.sumElems(G)[0] / ((width-2) * (height-2))

#
def calculate_variance(img1):
    # 确保图像为单通道
    if len(img1.shape) > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # 计算均值
    mean_val = cv2.mean(img1)[0]

    # 向量化计算方差（避免循环）
    img_float = img1.astype(np.float32)
    variance = np.sum(np.square(img_float - mean_val))

     # 新增归一化处理
    if mean_val == 0:
        return 0.0
    clarity_value = variance / mean_val  # 添加的归一化计算
    return clarity_value

    # return variance

if __name__ == '__main__':


# 测试图像
#     image = cv2.imread("window/control/auto/2025-03-17-17-00-32.bmp")
#     image2 = cv2.imread("window/control/auto/2025-03-17-17-01-17.bmp")
    image = cv2.imread("window/DA3562117.png")
    image2 = cv2.imread("window/DA3827093.png")

    # Tenengrad清晰度计算
    clarity_tenengrad = tenengrad(image)
    clarity_tenengrad2 = tenengrad(image2)
    print(f"Tenengrad清晰度: {clarity_tenengrad:.2f}")
    print(f"Tenengrad清晰度: {clarity_tenengrad2:.2f}")

    # 方差计算
    clarity_variance = calculate_variance(image)
    print(f"方差清晰度: {clarity_variance:.2f}")
    clarity_variance2 = calculate_variance(image2)
    print(f"方差清晰度: {clarity_variance2:.2f}")