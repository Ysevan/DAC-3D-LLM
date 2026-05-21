import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import font_manager


def tenengrad(image, roi=None, roi_type='rect'):
    """计算图像的 Tenengrad 清晰度（支持矩形ROI）"""
    assert image is not None, "Image is empty"

    # 转换为灰度图
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image.copy()

    # # 矩形ROI处理（固定为(1154,564)到(1215,617)）
    # if roi_type == 'rect':
    #     x1, y1, x2, y2 = 1154, 564, 1215, 617
    #     gray_img = gray_img[y1:y2, x1:x2]
    # # 圆形ROI处理（中心(1024,1024)，半径1000）
    # if roi_type == 'circle':
    #     center = (1024, 1024)
    #     radius = 1000
    #     # 创建圆形掩膜
    #     mask = np.zeros_like(gray_img)
    #     cv2.circle(mask, center, radius, 255, -1)
    #         # 应用掩膜提取ROI
    #     gray_img = cv2.bitwise_and(gray_img, gray_img, mask=mask)

    # 计算Sobel梯度
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度幅值
    G = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    # 计算均值（排除边缘像素）
    height, width = gray_img.shape
    return cv2.sumElems(G)[0] / ((width - 2) * (height - 2))


def laplacian_gradient(image, roi=None, roi_type='circle'):
    """计算图像的 Laplacian 清晰度（支持圆形ROI）"""
    assert image is not None, "Image is empty"

    # 转换为灰度图
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image.copy()

    # # 圆形ROI处理（中心(1024,1024)，半径1000）
    # if roi_type == 'circle':
    #     center = (1024, 1024)
    #     radius = 1000
    #     # 创建圆形掩膜
    #     mask = np.zeros_like(gray_img)
    #     cv2.circle(mask, center, radius, 255, -1)
    #     # 应用掩膜提取ROI
    #     gray_img = cv2.bitwise_and(gray_img, gray_img, mask=mask)

    # 计算Laplacian梯度
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)

    # 计算绝对值
    abs_laplacian = cv2.convertScaleAbs(laplacian)

    # 计算Laplacian梯度的均值作为清晰度指标
    height, width = gray_img.shape
    return cv2.sumElems(abs_laplacian)[0] / (width * height)

def get_values_from_folder(folder_path):
    """遍历文件夹，计算所有图片的两种清晰度指标"""
    tenengrad_values = []
    laplacian_values = []
    filenames = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)

            if image is not None:
                tenengrad_clarity = tenengrad(image)
                laplacian_clarity = laplacian_gradient(image)
                tenengrad_values.append(tenengrad_clarity)
                laplacian_values.append(laplacian_clarity)
                filenames.append(filename)
            else:
                print(f"⚠️ 无法读取图片: {filename}")

    return filenames, tenengrad_values, laplacian_values


def plot_dual_metrics(filenames, tenengrad_values, laplacian_values):
    """绘制两种清晰度指标的对比图"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 生成Z轴距离：从3.851开始，每张图片增加0.001
    z_distances = [2.38 + i * 0.001 for i in range(len(filenames))]

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 7))

    # 绘制Tenengrad曲线
    line1, = ax.plot(z_distances, tenengrad_values, 'bo-', label='Tenengrad 清晰度')

    # 绘制Laplacian曲线
    line2, = ax.plot(z_distances, laplacian_values, 'gs-', label='Laplacian 清晰度')

    # 找到并标注两种指标的最大值
    max_tenengrad = max(tenengrad_values)
    max_tenengrad_idx = tenengrad_values.index(max_tenengrad)
    max_tenengrad_z = z_distances[max_tenengrad_idx]

    max_laplacian = max(laplacian_values)
    max_laplacian_idx = laplacian_values.index(max_laplacian)
    max_laplacian_z = z_distances[max_laplacian_idx]

    # 标注Tenengrad最大值
    ax.plot(max_tenengrad_z, max_tenengrad, 'ro', markersize=10)
    ax.annotate(
        f'Tenengrad Max: {max_tenengrad:.2f}\nZ = {max_tenengrad_z:.3f} mm',
        xy=(max_tenengrad_z, max_tenengrad),
        xytext=(10, 20),
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->')
    )

    # 标注Laplacian最大值
    ax.plot(max_laplacian_z, max_laplacian, 'ro', markersize=10)
    ax.annotate(
        f'Laplacian Max: {max_laplacian:.2f}\nZ = {max_laplacian_z:.3f} mm',
        xy=(max_laplacian_z, max_laplacian),
        xytext=(10, -30),
        textcoords='offset points',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle='->')
    )

    # 图表装饰
    ax.set_title("两种清晰度指标对比（矩形ROI 1154,564-1215,617）")
    ax.set_xlabel("Z轴距离 (mm)")
    ax.set_ylabel("清晰度值")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()

    # 打印最大值信息
    print("\n🔍 最大值信息:")
    print(f"Tenengrad - Z轴距离: {max_tenengrad_z:.3f} mm, 清晰度: {max_tenengrad:.2f}")
    print(f"Laplacian - Z轴距离: {max_laplacian_z:.3f} mm, 清晰度: {max_laplacian:.2f}")


if __name__ == '__main__':
    # 指定文件夹路径
    folder_path = r"E:\ftkpic\layerscan\2025-04-21\11-26-02"  # 替换为你的文件夹路径

    # 获取所有图片的两种清晰度值
    filenames, tenengrad_values, laplacian_values = get_values_from_folder(folder_path)

    # 打印所有结果
    print("📊 清晰度随Z轴距离变化（矩形ROI 1154,564-1215,617）:")
    for i, (filename, t_val, l_val) in enumerate(zip(filenames, tenengrad_values, laplacian_values)):
        z_distance =2.38 + i * 0.001
        print(f"Z = {z_distance:.3f} mm | {filename}: Tenengrad={t_val:.2f}, Laplacian={l_val:.2f}")

    # 绘制对比图
    plot_dual_metrics(filenames, tenengrad_values, laplacian_values)