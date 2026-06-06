import cv2
import numpy as np
import os
import time
import concurrent.futures
from pathlib import Path
from homography_loader import load_homography_matrices, get_homographies_list  # 导入配置加载工具

# 检查CUDA是否可用
use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
print(f"GPU加速状态: {'启用' if use_gpu else '未启用'}")

# 创建结果目录
result_dir = r"C:\Users\Administrator\Desktop\yolov11\ultralytics\Registration_ImgFusion\results"
os.makedirs(result_dir, exist_ok=True)

###########################################
# 第一部分: 图像配准相关函数
###########################################

# 亮度标准化
def normalize_brightness(images):
    """标准化多张图片的亮度"""
    # 转换为灰度图计算亮度
    grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img for img in images]

    # 计算参考图像（中间图像）的统计值
    reference_idx = 1
    mean_ref, std_ref = cv2.meanStdDev(grays[reference_idx])
    mean_ref = mean_ref[0][0]
    std_ref = std_ref[0][0]

    # 标准化每张图像
    normalized_images = []
    for i, img in enumerate(images):
        if i == reference_idx:
            normalized_images.append(img)
        else:
            # 计算当前图像的统计值
            mean_cur, std_cur = cv2.meanStdDev(grays[i])
            mean_cur = mean_cur[0][0]
            std_cur = std_cur[0][0]

            # 计算亮度调整参数
            alpha = std_ref / (std_cur + 1e-6)  # 避免除以零
            beta = mean_ref - (mean_cur * alpha)

            # 调整图像亮度
            normalized = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            normalized_images.append(normalized)

    # 确保图像顺序正确：[焦后, 焦面, 焦前]
    if len(normalized_images) == 2:  # 如果只有两张调整后的图像
        return [normalized_images[0], images[reference_idx], normalized_images[1]]

    return normalized_images


def load_images(image_paths):
    """加载图像并进行亮度标准化"""
    images = []

    # 检查输入是否已经是图像数组
    if isinstance(image_paths[0], np.ndarray):
        # 如果输入已经是图像数组，直接处理
        original_images = image_paths
        normalized_images = normalize_brightness(original_images)

        # 重新生成灰度图
        normalized_pairs = [(img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img)
                            for img in normalized_images]

        return normalized_pairs

    # 使用串行处理加载图像
    def load_single_image(path):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法加载图像: {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return (img, gray)

    # 串行加载图像
    for path in image_paths:
        images.append(load_single_image(path))

    # 提取原始图像并进行亮度标准化
    original_images = [img for img, _ in images]
    normalized_images = normalize_brightness(original_images)

    # 重新生成灰度图
    normalized_pairs = [(img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for img in normalized_images]

    return normalized_pairs


def detect_features(gray_images, scale_factor=0.75, max_features=1000):
    """使用SIFT检测特征点和描述符"""
    keypoints_list = []
    descriptors_list = []

    # 缩放图像以加速处理
    scaled_grays = []
    for gray in gray_images:
        if scale_factor < 1.0:
            h, w = gray.shape
            scaled_h, scaled_w = int(h * scale_factor), int(w * scale_factor)
            scaled_gray = cv2.resize(gray, (scaled_w, scaled_h))
            scaled_grays.append(scaled_gray)
        else:
            scaled_grays.append(gray)

    # 创建SIFT检测器，添加参数限制特征点数量
    if use_gpu:
        # 使用GPU加速的SIFT，设置特征点数量上限
        sift = cv2.cuda_SIFT.SIFT_create(nfeatures=max_features)

        # 使用并行处理
        def process_gpu_image(gray):
            # 转换为GPU图像
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(gray)

            # 检测关键点和描述符
            keypoints, descriptors = sift.detectAndCompute(gpu_img, None)
            return keypoints, descriptors.download()

        # 并行处理GPU特征提取
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_gpu_image, scaled_grays))

        for keypoints, descriptors in results:
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)
    else:
        # 使用CPU版本的SIFT，设置特征点数量上限
        sift = cv2.SIFT_create(nfeatures=max_features)

        # 并行处理每张图像的特征提取
        def process_image(gray):
            return sift.detectAndCompute(gray, None)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_image, scaled_grays))

        for keypoints, descriptors in results:
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)

    # 如果使用了缩放，需要调整关键点坐标
    if scale_factor < 1.0:
        for kps in keypoints_list:
            for kp in kps:
                kp.pt = (kp.pt[0] / scale_factor, kp.pt[1] / scale_factor)

    return keypoints_list, descriptors_list

def match_features(descriptors1, descriptors2):
    """匹配两组特征点"""
    # 使用FLANN匹配器进行特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=32)  # 减少检查次数以提高速度

    if use_gpu:
        # GPU版本的特征匹配
        matcher = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_L2)

        # 上传描述符到GPU
        gpu_desc1 = cv2.cuda_GpuMat()
        gpu_desc2 = cv2.cuda_GpuMat()
        gpu_desc1.upload(np.float32(descriptors1))
        gpu_desc2.upload(np.float32(descriptors2))

        # 执行KNN匹配
        gpu_matches = matcher.knnMatch(gpu_desc1, gpu_desc2, k=2)

        # 下载匹配结果
        matches = []
        for match in gpu_matches:
            matches.append([match[0].download(), match[1].download()])
    else:
        # CPU版本的特征匹配
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 应用Lowe's比率测试筛选好的匹配点，调整比率阈值
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:  # 稍微放宽比率阈值
            good_matches.append(m)

    # 如果匹配点太多，只保留最好的前N个
    if len(good_matches) > 100:
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:100]

    return good_matches

def compute_homography(keypoints1, keypoints2, matches):
    """计算两组匹配点之间的单应性矩阵"""
    if len(matches) < 4:
        raise ValueError(f"匹配点数量不足，无法计算单应性矩阵。当前匹配点数: {len(matches)}")

    # 提取匹配点的坐标
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 计算内点数量
    inliers_count = np.sum(mask)

    return H, mask, inliers_count


def warp_image(img, H, target_shape):
    use_gpu = False
    h_target, w_target = target_shape
    dsize = (w_target, h_target)

    if use_gpu:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        warped = cv2.cuda.warpPerspective(
            gpu_img, H, dsize,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE  # 边缘自动延伸，无黑边无白边
        ).download()
    else:
        warped = cv2.warpPerspective(
            img, H, dsize,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE  # 关键
        )

    return warped
    # 优化1: 直接在warpPerspective中使用白色边界，避免后处理
    # if use_gpu:
    #     gpu_img = cv2.cuda_GpuMat()
    #     gpu_img.upload(img)
    #     warped = cv2.cuda.warpPerspective(gpu_img, H, (target_shape[1], target_shape[0]),
    #                                     flags=cv2.INTER_LINEAR,
    #                                     borderMode=cv2.BORDER_CONSTANT,
    #                                     borderValue=(0, 0, 0)).download()
    # else:
    #     warped = cv2.warpPerspective(img, H, (target_shape[1], target_shape[0]),
    #                                flags=cv2.INTER_LINEAR,
    #                                borderMode=cv2.BORDER_CONSTANT,
    #                                borderValue=(0, 0, 0))
    #
    # # 优化2: 简化黑色区域检测和替换，避免多次数组操作
    # if len(warped.shape) == 3:
    #     # 对于彩色图像，找出所有通道都为0的像素
    #     black_mask = np.all(warped == 0, axis=0)
    #     warped[black_mask] = 0
    # else:
    #     # 对于灰度图像
    #     black_mask = warped == 0
    #     warped[black_mask] = 0
    #
    # return warped

def register_images(images, reference_idx=1):
    """对图像进行配准，以reference_idx指定的图像为参考"""
    # 提取灰度图像
    gray_images = [gray for _, gray in images]

    # 检测特征点
    print("正在检测特征点...")
    feature_start_time = time.time()
    keypoints_list, descriptors_list = detect_features(gray_images)
    feature_end_time = time.time()
    print(f"特征点检测完成，共检测到 {[len(kp) for kp in keypoints_list]} 个特征点")
    print(f"特征点检测耗时: {feature_end_time - feature_start_time:.4f} 秒")

    # 预分配homographies列表
    homographies = [None] * len(images)
    homographies[reference_idx] = np.eye(3)  # 参考图像到自身的变换是单位矩阵

    # 计算每张图像到参考图像的变换矩阵
    matching_start_time = time.time()

    # 定义处理单个图像的函数
    def process_image(i):
        if i == reference_idx:
            return None  # 参考图像已处理

        print(f"正在计算图像 {i+1} 到参考图像的变换矩阵...")

        # 匹配特征点
        matches = match_features(descriptors_list[i], descriptors_list[reference_idx])
        print(f"找到 {len(matches)} 个匹配点")

        # 计算单应性矩阵
        H, mask, inliers_count = compute_homography(
            keypoints_list[i], keypoints_list[reference_idx], matches
        )
        print(f"内点数量: {inliers_count}")

        return (i, H)

    # 并行处理图像
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, i) for i in range(len(images)) if i != reference_idx]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                i, H = result
                homographies[i] = H

    matching_end_time = time.time()
    print(f"特征匹配和变换计算总耗时: {matching_end_time - matching_start_time:.4f} 秒")

    # 变换图像
    warped_images = []
    ref_shape = images[reference_idx][0].shape

    for i, (img, _) in enumerate(images):
        if i == reference_idx:
            warped_images.append(img)
        else:
            warped = warp_image(img, homographies[i], ref_shape)
            warped_images.append(warped)

    return warped_images, homographies

###########################################
# 第二部分: 图像融合相关函数
###########################################

def adjust_brightness_contrast(img, alpha, beta):
    """调整图像的亮度和对比度"""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def compute_gradient(img):
    """使用Sobel算子计算梯度（高性能优化版本）"""
    # 优化：使用Sobel 3x3核（比Scharr更快，质量差别不大）
    grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3)
    # 使用绝对值和加快速度（比magnitude稍快）
    gradient = cv2.convertScaleAbs(grad_x) + cv2.convertScaleAbs(grad_y)
    return gradient.astype(np.float32)

def compute_window_gradients_sum(gradient):
    """使用卷积操作计算窗口梯度和（高性能优化版本）"""
    # 优化1: 减小窗口从6x6到4x4（速度提升约2倍，质量影响小）
    window_size = 4

    # 优化2: 使用boxFilter替代filter2D（OpenCV内部优化，更快）
    result = cv2.boxFilter(gradient, -1, (window_size, window_size), normalize=False)
    return result

def process_image_for_fusion(img, reference_mean, reference_stddev):
    """处理单个图像的完整流程"""
    # 计算并调整图像
    mean, stddev = cv2.meanStdDev(img)
    mean_val = mean[0][0]
    stddev_val = stddev[0][0]
    alpha = reference_stddev / stddev_val
    beta = reference_mean - (mean_val * alpha)
    img_adjusted = adjust_brightness_contrast(img, alpha, beta)

    # 计算梯度
    grad = compute_gradient(img_adjusted)

    # 计算窗口梯度和
    grad_sum = compute_window_gradients_sum(grad)

    return img_adjusted, grad_sum

def fuse_images(grad_sum1, grad_sum2, grad_sum3, img1, img2, img3):
    """优化的图像融合函数，使用加权融合并处理无效值（性能优化版）"""
    # 优化1: 使用更快的核大小(3x3代替5x5)并并行处理高斯模糊
    # 使用栈操作一次性处理所有梯度图，减少函数调用开销
    grad_stack = np.stack([grad_sum1, grad_sum2, grad_sum3], axis=0)

    # 对每个梯度图进行高斯模糊（使用更小的核以提速）
    blurred_grads = []
    for grad in grad_stack:
        blurred_grads.append(cv2.GaussianBlur(grad, (3, 3), 0))

    grad_sum1, grad_sum2, grad_sum3 = blurred_grads[0], blurred_grads[1], blurred_grads[2]

    # 确保梯度值非负（向量化操作）
    grad_sum1 = np.maximum(grad_sum1, 0)
    grad_sum2 = np.maximum(grad_sum2, 0)
    grad_sum3 = np.maximum(grad_sum3, 0)

    # 计算总和，添加小的epsilon值避免除以0
    epsilon = 1e-10
    total = grad_sum1 + grad_sum2 + grad_sum3 + epsilon

    # 优化2: 直接计算归一化权重，减少一次除法操作
    weight1 = grad_sum1 / total
    weight2 = grad_sum2 / total
    weight3 = grad_sum3 / total

    # 扩展权重维度以匹配图像通道
    if len(img1.shape) == 3:
        weight1 = weight1[:, :, np.newaxis]
        weight2 = weight2[:, :, np.newaxis]
        weight3 = weight3[:, :, np.newaxis]

    # 优化3: 直接使用uint8类型进行加权融合，避免float32的大内存开销
    # 先转为float32进行计算，但立即转回uint8
    fused_image = np.clip(
        img1.astype(np.float32) * weight1 +
        img2.astype(np.float32) * weight2 +
        img3.astype(np.float32) * weight3,
        0, 255
    ).astype(np.uint8)

    return fused_image


def fusion_process(warped_images, use_downscale=True, downscale_factor=0.5):
    """优化的融合处理函数，消除重复计算（高性能版本）

    参数:
        warped_images: 配准后的图像列表
        use_downscale: 是否使用降采样优化（默认True，可提升2-3倍速度）
        downscale_factor: 降采样比例（默认0.5，即缩小到50%计算权重）
    """
    # 步骤1: 转换为灰度图
    gray_images = []
    for img in warped_images:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        gray_images.append(gray)

    # 步骤2: 计算参考图像的统计值
    reference_idx = 1
    mean, stddev = cv2.meanStdDev(gray_images[reference_idx])
    reference_mean = mean[0][0]
    reference_stddev = stddev[0][0]

    # 步骤3: 调整亮度对比度（在计算梯度之前）
    adjusted_images = []
    for i, gray in enumerate(gray_images):
        if i == reference_idx:
            adjusted_images.append(gray)
        else:
            mean_cur, stddev_cur = cv2.meanStdDev(gray)
            mean_val = mean_cur[0][0]
            stddev_val = stddev_cur[0][0]
            alpha = reference_stddev / (stddev_val + 1e-6)
            beta = reference_mean - (mean_val * alpha)
            adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            adjusted_images.append(adjusted)

    # 步骤4: 计算梯度权重（可选降采样优化）
    if use_downscale and downscale_factor < 1.0:
        # 降采样计算梯度权重（速度提升2-3倍）
        h, w = adjusted_images[0].shape
        new_h, new_w = int(h * downscale_factor), int(w * downscale_factor)

        # 缩小图像
        small_images = [cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                       for img in adjusted_images]

        # 在小图上计算梯度
        small_grad_sums = []
        for small_img in small_images:
            gradient = compute_gradient(small_img)
            grad_sum = compute_window_gradients_sum(gradient)
            small_grad_sums.append(grad_sum)

        # 上采样权重图回到原始大小
        grad_sums = [cv2.resize(grad, (w, h), interpolation=cv2.INTER_LINEAR)
                    for grad in small_grad_sums]
    else:
        # 原始尺寸计算
        grad_sums = []
        for adjusted_img in adjusted_images:
            gradient = compute_gradient(adjusted_img)
            grad_sum = compute_window_gradients_sum(gradient)
            grad_sums.append(grad_sum)

    # 步骤5: 融合图像
    fused = fuse_images(grad_sums[0], grad_sums[1], grad_sums[2],
                        warped_images[0], warped_images[1], warped_images[2])

    # 步骤6: 简化掩码处理
    if len(fused.shape) == 3:
        black_mask = np.all(fused < 10, axis=2)
        fused[black_mask] = 255
    else:
        black_mask = fused < 10
        fused[black_mask] = 255

    return fused

def register_and_fuse(image_paths, reference_idx=1):
    """完整的图像配准和融合流程"""
    total_start_time = time.time()

    # 加载图像
    print("正在加载图像...")
    load_start_time = time.time()
    images = load_images(image_paths)
    load_end_time = time.time()
    print(f"图像加载完成，共 {len(images)} 张图像")
    print(f"图像加载耗时: {load_end_time - load_start_time:.4f} 秒")

    # 图像配准
    print("\n开始图像配准...")
    registration_start_time = time.time()
    warped_images, homographies = register_images(images, reference_idx)
    registration_end_time = time.time()
    print(f"图像配准完成")
    print(f"图像配准总耗时: {registration_end_time - registration_start_time:.4f} 秒")

    # 保存配准后的图像
    for i, img in enumerate(warped_images):
        warped_path = os.path.join(result_dir, f"warped_{i+1}.jpg")
        cv2.imwrite(warped_path, img)

    # 图像融合
    print("\n开始图像融合...")
    fusion_start_time = time.time()
    fused_image = fusion_process(warped_images)
    fusion_end_time = time.time()
    print(f"图像融合完成")
    print(f"图像融合总耗时: {fusion_end_time - fusion_start_time:.4f} 秒")

    # 保存融合后的图像
    fused_path = os.path.join(result_dir, "fused_image.jpg")
    cv2.imwrite(fused_path, fused_image)

    # 计算总执行时间
    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    # 打印时间统计信息
    print("\n时间统计:")
    print(f"总执行时间: {total_time:.4f} 秒")
    print(f"图像加载时间: {load_end_time - load_start_time:.4f} 秒 ({(load_end_time - load_start_time) / total_time * 100:.2f}%)")
    print(f"图像配准时间: {registration_end_time - registration_start_time:.4f} 秒 ({(registration_end_time - registration_start_time) / total_time * 100:.2f}%)")
    print(f"图像融合时间: {fusion_end_time - fusion_start_time:.4f} 秒 ({(fusion_end_time - fusion_start_time) / total_time * 100:.2f}%)")

    # 保存性能报告到文本文件
    performance_path = os.path.join(result_dir, "performance_report.txt")
    with open(performance_path, 'w', encoding='utf-8') as f:
        f.write(f"GPU加速状态: {'启用' if use_gpu else '未启用'}\n")
        f.write(f"总执行时间: {total_time:.4f} 秒\n")
        f.write(f"图像加载时间: {load_end_time - load_start_time:.4f} 秒 ({(load_end_time - load_start_time) / total_time * 100:.2f}%)\n")
        f.write(f"图像配准时间: {registration_end_time - registration_start_time:.4f} 秒 ({(registration_end_time - registration_start_time) / total_time * 100:.2f}%)\n")
        f.write(f"图像融合时间: {fusion_end_time - fusion_start_time:.4f} 秒 ({(fusion_end_time - fusion_start_time) / total_time * 100:.2f}%)\n")

    print(f"性能报告已保存到: {performance_path}")
    print(f"所有结果已保存到: {result_dir}")

    return fused_image, warped_images, homographies

#这里需要将计算出的矩阵填写到里面去
def fusion_with_known_matrices(image_paths, homographies=None, result_dir=None, verbose=True,
                              use_downscale=True, downscale_factor=0.5):
    """
    使用已知变换矩阵进行图像融合

    参数:
        image_paths: 三张图像的路径列表 [焦后, 焦面, 焦前]
        homographies: 预先确定的变换矩阵列表，如果为None则使用默认值
        result_dir: 结果保存目录
        verbose: 是否打印详细信息
        use_downscale: 是否使用降采样优化（默认True，可提升2-3倍速度）
        downscale_factor: 降采样比例（默认0.5，建议0.4-0.6之间）
    """
    if result_dir is None:
        result_dir = r"D:\zycgit\ZDevelop_Confocal\xxp_ui\results\fusion_result"
    os.makedirs(result_dir, exist_ok=True)

    # 如果没有提供变换矩阵，从配置文件自动加载
    if homographies is None:
        print("未提供变换矩阵，正在从配置文件加载...")
        H1_to_ref, H3_to_ref = load_homography_matrices()
        homographies = [H1_to_ref, np.eye(3), H3_to_ref]  # 参考图像使用单位矩阵
        print("旋转矩阵已从配置文件加载")

    # 加载图像
    if verbose:
        print("正在加载图像...")
    images = load_images(image_paths)

    # # 保存标准化后的原始图像
    # for i, (img, _) in enumerate(images):
    #     orig_path = os.path.join(result_dir, f"normalized_{i+1}.jpg")
    #     cv2.imwrite(orig_path, img)
    #     if verbose:
    #         print(f"标准化图像已保存: {orig_path}")

    # 变换图像
    reference_idx = 1  # 焦面图像索引
    warped_images = []
    for i, (img, _) in enumerate(images):
        if i == reference_idx:
            warped_images.append(img)  # 参考图像不需要变换
        else:
            warped = warp_image(img, homographies[i], images[reference_idx][0].shape)
            warped_images.append(warped)

    # 保存配准后的图像
    for i, img in enumerate(warped_images):
        warped_path = os.path.join(result_dir, f"warped_{i+1}.jpg")
        cv2.imwrite(warped_path, img)
        if verbose:
            print(f"配准图像已保存: {warped_path}")

    # 图像融合
    if verbose:
        print("\n开始图像融合...")
    fused_image = fusion_process(warped_images, use_downscale=use_downscale,
                                 downscale_factor=downscale_factor)

    # 保存融合后的图像
    fused_path = os.path.join(result_dir, "fused_image.jpg")
    cv2.imwrite(fused_path, fused_image)
    if verbose:
        print(f"融合图像已保存: {fused_path}")

    return fused_image, warped_images




# 修改main函数
def main():
    """主函数"""
    # 设置图像路径
    image_dir = r"D:\zycgit\ZDevelop_Confocal\xxp_ui\Algorithm\demo_img\demo"
    image_paths = [
        os.path.join(image_dir, "焦后.bmp"),  # 焦后图像
        os.path.join(image_dir, "焦面.bmp"),  # 焦面图像（参考）
        os.path.join(image_dir, "焦前.bmp")   # 焦前图像
    ]

    # 自动从配置文件加载变换矩阵
    print("正在从配置文件加载旋转矩阵...")
    H1_to_ref, H3_to_ref = load_homography_matrices()
    homographies = [H1_to_ref, np.eye(3), H3_to_ref]

    print("\n加载的变换矩阵:")
    print("H1_to_ref (焦后到焦面):")
    print(H1_to_ref)
    print("\nH3_to_ref (焦前到焦面):")
    print(H3_to_ref)

    # 执行融合
    fused_image, warped_images = fusion_with_known_matrices(
        image_paths,
        homographies=homographies,
        verbose=True,
        use_downscale = False
    )

    # 显示结果
    try:
        # cv2.imshow("配准后的焦后图像", warped_images[0])
        # cv2.imshow("参考图像", warped_images[1])
        # cv2.imshow("配准后的焦前图像", warped_images[2])
        # cv2.imshow("融合结果", fused_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("无法显示图像，但结果已保存到磁盘")

if __name__ == "__main__":
    # 创建图像目录
    image_dir = r"D:\zycgit\ZDevelop_Confocal\xxp_ui\Algorithm\demo_img"
    os.makedirs(image_dir, exist_ok=True)

    main()

