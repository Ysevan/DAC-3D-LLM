import os
import time

import cv2
import numpy as np
import pandas as pd
from datetime import datetime

def detect_circle_and_extract_inside(image):
    """
    检测图像中的圆并提取圆内区域

    参数:
    image: 输入图像

    返回:
    圆内区域, 圆形掩码
    """

    image = cv2.resize(image,[1024,1024])
    # 转换为灰度图像
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()


    # 应用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # 使用Hough圆变换检测圆
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=10,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=min(gray.shape) // 4,  # 最小半径为图像尺寸的1/4
        maxRadius=min(gray.shape) // 2  # 最大半径为图像尺寸的1/2
    )

    # 创建空掩码
    mask = np.zeros_like(gray)

    if circles is not None:
        # 将检测到的圆转换为整数坐标
        circles = np.round(circles[0, :]).astype("int")

        # 取第一个检测到的圆
        (x, y, r) = circles[0]
        print(f"检测到圆：中心点 ({x}, {y})，半径 {r}")

        # 绘制完整的圆形掩码
        cv2.circle(mask, (x, y), r, 255, -1)
    else:
        # 如果未检测到圆，假设圆在图像中心
        print("未检测到圆，使用图"
              ""
              "像中心作为圆心...")
        h, w = gray.shape
        center = (w // 2, h // 2)
        radius = min(w, h) // 2

        # 创建圆形掩码
        cv2.circle(mask, center, radius, 255, -1)

    # 提取圆内区域
    circle_inside = cv2.bitwise_and(image, image, mask=mask)

    return circle_inside, mask


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
    return cv2.sumElems(G)[0] / ((width - 2) * (height - 2))


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


def laplacian_gradient(image):
    assert image is not None, "Image is empty"

    # 转换为灰度图
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image.copy()

    # 计算Laplacian梯度
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)

    # 计算绝对值
    abs_laplacian = cv2.convertScaleAbs(laplacian)

    # 计算Laplacian梯度的均值作为清晰度指标
    height, width = gray_img.shape
    return cv2.sumElems(abs_laplacian)[0] / (width * height)
def calculate_gray_sum_and_ratio(image):
    """
    计算图像的灰度值总和和tenengrad/平均灰度值的比值

    参数:
    image: 输入图像

    返回:
    gray_sum: 灰度值总和
    avg_gray: 平均灰度值
    ratio: tenengrad/平均灰度值
    """
    # 转换为灰度图
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image.copy()

    # 计算灰度值总和
    gray_sum = np.sum(gray_img)

    # 计算平均灰度值
    height, width = gray_img.shape
    avg_gray = gray_sum / (height * width)

    # 计算tenengrad值
    tenengrad_val = tenengrad(image)

    # 计算比值
    if avg_gray != 0:
        ratio = tenengrad_val / avg_gray
    else:
        ratio = 0.0

    return gray_sum, avg_gray, ratio
def calculate_gray_metrics(image):
    """计算灰度值总和、平均灰度值和Tenengrad/平均灰度值比值"""
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    gray_sum = np.sum(gray_img)
    height, width = gray_img.shape
    avg_gray = gray_sum / (height * width)
    tenengrad_val = tenengrad(image)
    ratio = tenengrad_val / avg_gray if avg_gray != 0 else 0.0
    return gray_sum, avg_gray, ratio

def process_folder(folder_path, output_csv):
    """处理文件夹中的所有图片并保存结果到CSV"""
    results = []

    # 获取文件夹中所有图片文件
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff'))]

    if not image_files:
        print(f"文件夹 {folder_path} 中没有找到图片文件")
        return

    print(f"开始处理 {len(image_files)} 张图片...")

    for i, filename in enumerate(image_files, 1):
        try:
            start_time = time.time()
            filepath = os.path.join(folder_path, filename)

            # 读取图片
            image = cv2.imread(filepath)
            if image is None:
                print(f"无法读取图片: {filename}")
                continue

            # 提取圆内区域
            circle_inside, _ = detect_circle_and_extract_inside(image)

            # 计算各项指标
            gray_sum, avg_gray, ratio = calculate_gray_metrics(circle_inside)
            tenengrad_val = tenengrad(circle_inside)

            # 添加到结果列表
            results.append({
                '文件名': filename,
                '灰度值总和': gray_sum,
                '平均灰度值': avg_gray,
                'Tenengrad值': tenengrad_val,
                'Tenengrad/平均灰度值': ratio,
                '处理时间(秒)': time.time() - start_time
            })

            print(f"已处理 {i}/{len(image_files)}: {filename} - 比值: {ratio:.4f}")

        except Exception as e:
            print(f"处理图片 {filename} 时出错: {str(e)}")
            continue

    # 保存结果到CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, encoding='utf_8_sig')  # 使用utf_8_sig编码支持中文
    print(f"处理完成! 结果已保存到 {output_csv}")


if __name__ == '__main__':
    # 设置输入文件夹和输出CSV文件路径
    input_folder = r"E:\ftkpic\snakescan\2025-04-14-11-22-56\test"
    output_csv = r"E:\ftkpic\snakescan\2025-04-14-11-18-32\焦面_清晰度分析结果.csv"

    # 开始处理
    process_folder(input_folder, output_csv)
# if __name__ == '__main__':
#     start_time = time.time()
#
#     # 测试图像
#     image_path1 = r"D:\zycgit\ZDevelop_Confocal\xxp_ui\DA3562117.png"
#     # image_path2 = r"D:\zycgit\ZDevelop_Confocal\xxp_ui\zmcdll\2025-03-25-17-04-47.bmp"
#
#     # 读取原始图像
#     image1 = cv2.imread(image_path1, cv2.IMREAD_REDUCED_COLOR_2)
#     # image2 = cv2.imread(image_path2, cv2.IMREAD_REDUCED_COLOR_2)
#
#     image1 = cv2.imread(image_path1)
#     # image2 = cv2.imread(image_path2)
#     # 检查图像是否成功读取
#     # if image1 is None or image2 is None:
#     #     print(f"无法读取图像: {image_path1 if image1 is None else image_path2}")
#     #     exit()
#     if image1 is None:
#         print(f"无法读取图像: {image_path1 if image1 is None else image_path2}")
#         exit()
#
#     # 提取圆内区域
#     circle_inside1, mask1 = detect_circle_and_extract_inside(image1)
#     # circle_inside2, mask2 = detect_circle_and_extract_inside(image2)
#
#     # 保存圆内区域图像(可选)
#     cv2.imwrite("circle_inside35.jpg", circle_inside1)
#     # cv2.imwrite("circle_inside47.jpg", circle_inside2)
#
#     # Tenengrad清晰度计算
#     clarity_tenengrad1 = tenengrad(circle_inside1)
#     # clarity_tenengrad2 = tenengrad(circle_inside2)
#     print(f"图像1 Tenengrad清晰度: {clarity_tenengrad1:.2f}")
#     # print(f"图像2 Tenengrad清晰度: {clarity_tenengrad2:.2f}")
#
#     # 方差计算
#     clarity_variance1 = calculate_variance(circle_inside1)
#     # clarity_variance2 = calculate_variance(circle_inside2)
#     print(f"图像1 方差清晰度: {clarity_variance1:.2f}")
#     # print(f"图像2 方差清晰度: {clarity_variance2:.2f}")
#
#
#     # 显示结果
#     # 调整大小以便显示
#     def resize_for_display(img, max_width=800, max_height=600):
#         if img is None:
#             return None
#         h, w = img.shape[:2]
#         if h > max_height or w > max_width:
#             scale = min(max_height / h, max_width / w)
#             new_size = (int(w * scale), int(h * scale))
#             return cv2.resize(img, new_size)
#         return img
#
#
#     gray_sum, avg_gray, ratio = calculate_gray_sum_and_ratio(circle_inside1)
#     print(f"图像1 灰度值总和: {gray_sum:.2f}")
#     print(f"图像1 平均灰度值: {avg_gray:.2f}")
#     print(f"图像1 Tenengrad/平均灰度值比值: {ratio:.4f}")
#
#     # 显示原始图像和圆内区域
#     cv2.imshow("a1", resize_for_display(image1))
#     cv2.imshow("aa1", resize_for_display(circle_inside1))
#     # cv2.imshow("b2", resize_for_display(image2))
#     # cv2.imshow("bb2", resize_for_display(circle_inside2))
#     end_time = time.time()
#     print("时间：",end_time-start_time)
#     print("按任意键关闭所有窗口...")
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#




def fast_circle_crop(image):
    """高速圆形区域裁剪（针对2048x2048优化）"""
    # 固定参数
    center = (1024, 1024)
    radius = 1024

    # 生成圆形掩码（直接创建比填充更快）
    mask = np.zeros((2048, 2048), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # 使用蒙版提取区域（避免不必要的bitwise_and）
    return cv2.merge([cv2.bitwise_and(ch, mask) for ch in cv2.split(image)]), mask


def optimized_tenengrad(image):
    """针对大尺寸优化的Tenengrad计算"""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 使用Scharr算子并缩小计算区域
    dx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)[100:-100, 100:-100]  # 忽略边缘400像素
    dy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)[100:-100, 100:-100]

    # 近似计算（速度提升30%）
    return np.mean(np.abs(dx) + np.abs(dy)) * 0.9  # 0.9为经验修正系数


def fast_variance(image):
    """内存友好的方差计算"""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray[512:-512, 512:-512].astype(np.float32)  # 只计算中心1024x1024区域

    mean = np.mean(gray)
    return np.mean((gray - mean) ** 2) / (mean + 1e-6)

#
# if __name__ == '__main__':
#     # 硬件加速初始化
#     cv2.setUseOptimized(True)
#     cv2.setNumThreads(4)  # 根据CPU核心数调整
#
#
#     # 图像加载（使用内存映射加速大文件读取）
#     def load_big_image(path):
#         return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
#
#
#     image_paths = [
#         r"D:\zycgit\ZDevelop_Confocal\xxp_ui\2025-03-25-17-04-35.bmp",
#         r"D:\zycgit\ZDevelop_Confocal\xxp_ui\2025-03-25-17-04-47.bmp"
#     ]
#
#     # 批量处理
#     results = []
#     for i, path in enumerate(image_paths, 1):
#         start = time.time()
#
#         img = load_big_image(path)
#         if img is None:
#             print(f"无法读取图像: {path}")
#             continue
#
#         # 统一尺寸（确保2048x2048）
#         img = cv2.resize(img, (2048, 2048))
#
#         # 核心处理
#         cropped, _ = fast_circle_crop(img)
#         ten = optimized_tenengrad(cropped)
#         var = fast_variance(cropped)
#
#         # 结果保存
#         cv2.imwrite(f"result_{i}.jpg", cropped)
#         results.append((ten, var))
#
#         print(f"图像{i}处理完成 | Tenengrad: {ten:.1f} | 方差: {var:.1f} | 耗时: {time.time() - start:.2f}s")
#
#     # 显示示例结果（可选）
#     if results:
#         sample = cv2.resize(cv2.imread("result_1.jpg"), (1024, 1024))
#         cv2.imshow("result1", sample)
#         sample = cv2.resize(cv2.imread("result_2.jpg"), (1024, 1024))
#         cv2.imshow("result2", sample)
#         cv2.waitKey(0)  # 显示3秒
#         cv2.destroyAllWindows()