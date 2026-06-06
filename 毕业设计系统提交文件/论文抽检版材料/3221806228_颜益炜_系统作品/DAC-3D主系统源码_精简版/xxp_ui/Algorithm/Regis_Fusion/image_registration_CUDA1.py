import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import concurrent.futures  # 添加并行处理支持
import json  # 添加JSON支持用于保存配置

# 检查CUDA是否可用
use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
print(f"GPU加速状态: {'启用' if use_gpu else '未启用'}")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False 

def load_images(image_paths):
    """加载图像并转换为灰度图"""
    images = []
    
    # 使用并行处理加载图像
    def load_single_image(path):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法加载图像: {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return (img, gray)
    
    # 使用线程池并行加载图像
    with concurrent.futures.ThreadPoolExecutor() as executor:
        images = list(executor.map(load_single_image, image_paths))
    
    return images

def detect_features(gray_images):
    """使用SIFT检测特征点和描述符"""
    keypoints_list = []
    descriptors_list = []
    
    # 创建SIFT检测器，添加参数限制特征点数量
    if use_gpu:
        # 使用GPU加速的SIFT，设置特征点数量上限
        sift = cv2.cuda_SIFT.SIFT_create(nfeatures=5000)  # 增加特征点数量以提高配准精度
        
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
            results = list(executor.map(process_gpu_image, gray_images))
            
        for keypoints, descriptors in results:
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)
    else:
        # 使用CPU版本的SIFT，设置特征点数量上限
        sift = cv2.SIFT_create(nfeatures=5000)  # 增加特征点数量以提高配准精度
        
        # 并行处理每张图像的特征提取
        def process_image(gray):
            return sift.detectAndCompute(gray, None)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_image, gray_images))
            
        for keypoints, descriptors in results:
            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)
    
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
        if m.distance < 0.7 * n.distance:  # 更严格的比率阈值以提高匹配质量
            good_matches.append(m)
    
    # 如果匹配点太多，只保留最好的前N个
    if len(good_matches) > 500:
        good_matches = sorted(good_matches, key=lambda x: x.distance)[:500]
    
    return good_matches

def compute_homography(keypoints1, keypoints2, matches):
    """计算两组匹配点之间的单应性矩阵"""
    if len(matches) < 4:
        raise ValueError(f"匹配点数量不足，无法计算单应性矩阵。当前匹配点数: {len(matches)}")

    # 提取匹配点的坐标
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

    # 计算内点数量
    inliers_count = np.sum(mask)

    # 验证单应性矩阵的合理性
    if H is not None:
        # 提取旋转和缩放信息
        a, b = H[0, 0], H[0, 1]
        c, d = H[1, 0], H[1, 1]

        # 计算旋转角度
        theta = np.arctan2(c, a) * 180 / np.pi

        # 计算缩放因子
        scale_x = np.sqrt(a**2 + c**2)
        scale_y = np.sqrt(b**2 + d**2)

        # 检查是否超出合理范围（对于显微镜焦平面图像，旋转应该很小，缩放接近1）
        if abs(theta) > 10:  # 旋转角度超过10度
            print(f"警告: 检测到异常旋转角度 {theta:.2f}°，可能配准失败")
        if scale_x < 0.8 or scale_x > 1.2 or scale_y < 0.8 or scale_y > 1.2:  # 缩放超过20%
            print(f"警告: 检测到异常缩放 (x: {scale_x:.2f}, y: {scale_y:.2f})，可能配准失败")
        if inliers_count < len(matches) * 0.3:  # 内点比例低于30%
            print(f"警告: 内点比例过低 ({inliers_count}/{len(matches)} = {inliers_count/len(matches)*100:.1f}%)，可能配准失败")

    return H, mask, inliers_count

def draw_matches(img1, kp1, img2, kp2, matches, mask=None):
    """绘制匹配结果"""
    match_img = cv2.drawMatches(
        img1, kp1, 
        img2, kp2, 
        matches, None, 
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=mask.ravel().tolist() if mask is not None else None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return match_img

def warp_image(img, H, target_shape):
    """使用单应性矩阵变换图像"""
    if use_gpu:
        # GPU版本的图像变换
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        
        # 创建输出GPU图像
        gpu_warped = cv2.cuda_GpuMat(target_shape[0], target_shape[1], gpu_img.type())
        
        # 执行变换
        gpu_warped = cv2.cuda.warpPerspective(gpu_img, H, (target_shape[1], target_shape[0]))
        
        # 下载结果
        return gpu_warped.download()
    else:
        # CPU版本的图像变换
        return cv2.warpPerspective(img, H, (target_shape[1], target_shape[0]))

def main():
    # 记录总执行时间
    total_start_time = time.time()
    
    # 图像路径 - 使用中文路径.
    image_paths = [
        r"D:\zycgit\ZDevelop_Confocal\xxp_ui\Algorithm\demo_img\demo\焦后.bmp",
        r"D:\zycgit\ZDevelop_Confocal\xxp_ui\Algorithm\demo_img\demo\焦面.bmp",
        r"D:\zycgit\ZDevelop_Confocal\xxp_ui\Algorithm\demo_img\demo\焦前.bmp"
    ]
    
    # 检查文件是否存在
    for path in image_paths:
        if not os.path.exists(path):
            print(f"警告: 文件不存在: {path}")
            return
    
    # 创建结果目录
    result_dir = r"D:\zycgit\ZDevelop_Confocal\xxp_ui\Algorithm\result"
    os.makedirs(result_dir, exist_ok=True)
    
    # 加载图像
    print("正在加载图像...")
    load_start_time = time.time()
    try:
        images = load_images(image_paths)
        load_end_time = time.time()
        print(f"图像加载成功! 耗时: {load_end_time - load_start_time:.4f} 秒")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    
    # 检测特征点
    print("正在检测特征点...")
    feature_start_time = time.time()

    # 直接使用原始分辨率以保证配准精度
    gray_images = [gray for _, gray in images]

    keypoints_list, descriptors_list = detect_features(gray_images)
    
    feature_end_time = time.time()
    print(f"特征点检测完成，共检测到 {[len(kp) for kp in keypoints_list]} 个特征点")
    print(f"特征点检测耗时: {feature_end_time - feature_start_time:.4f} 秒")

    
    
    # 以第二张图像(焦面)为参考，计算其他图像到参考图像的变换矩阵
    reference_idx = 1  # 焦面图像索引
    homographies = []
    
    # 计算每张图像到参考图像的变换矩阵
    matching_start_time = time.time()
    
    # 预分配homographies列表
    homographies = [None] * len(images)
    homographies[reference_idx] = np.eye(3)  # 参考图像到自身的变换是单位矩阵
    
    # 定义处理单个图像的函数
    def process_image(i):
        if i == reference_idx:
            return None  # 参考图像已处理
        
        print(f"正在计算图像 {i+1} 到参考图像的变换矩阵...")
        step_start_time = time.time()
        
        # 匹配特征点
        matches = match_features(descriptors_list[i], descriptors_list[reference_idx])
        print(f"找到 {len(matches)} 个匹配点")
        
        # 计算单应性矩阵
        H, mask, inliers_count = compute_homography(
            keypoints_list[i], keypoints_list[reference_idx], matches
        )
        print(f"内点数量: {inliers_count}")
        
        # 绘制匹配结果
        match_img = draw_matches(
            images[i][0], keypoints_list[i], 
            images[reference_idx][0], keypoints_list[reference_idx], 
            matches, mask
        )
        
        # 保存匹配结果
        match_path = os.path.join(result_dir, f"match_{i+1}_to_ref.jpg")
        cv2.imwrite(match_path, match_img)
        
        # 变换图像
        warped = warp_image(images[i][0], H, images[reference_idx][0].shape)
        
        # 保存变换后的图像
        warped_path = os.path.join(result_dir, f"warped_{i+1}_to_ref.jpg")
        cv2.imwrite(warped_path, warped)
        
        step_end_time = time.time()
        print(f"图像 {i+1} 处理耗时: {step_end_time - step_start_time:.4f} 秒")
        
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
    
        # ... 前面的代码保持不变 ...
    
    # 保存变换矩阵并计算旋转角度和平移距离
    for i, H in enumerate(homographies):
        if i == reference_idx:
            continue
        print(f"\n图像 {i+1} 到参考图像的变换矩阵:")
        print(H)
        
        # 计算旋转角度
        # 从单应性矩阵的左上角2x2子矩阵提取旋转信息
        a, b = H[0, 0], H[0, 1]
        c, d = H[1, 0], H[1, 1]
        
        # 计算旋转角度（两种方法）
        theta1 = np.arctan2(c, a) * 180 / np.pi
        theta2 = np.arctan2(-b, d) * 180 / np.pi
        
        # 取平均值作为最终角度
        rotation_angle = (theta1 + theta2) / 2 if abs(theta1 - theta2) < 10 else theta1
        
        # 提取平移向量
        tx, ty = H[0, 2], H[1, 2]
        
        # 计算总平移距离
        total_translation = np.sqrt(tx**2 + ty**2)
        
        # 输出计算结果
        image_names = ["焦后", "焦面", "焦前"]
        print(f"\n{image_names[i]}到{image_names[reference_idx]}的变换分析:")
        print(f"旋转角度: {rotation_angle:.2f}°")
        print(f"X方向平移: {tx:.2f} 像素")
        print(f"Y方向平移: {ty:.2f} 像素")
        print(f"总平移距离: {total_translation:.2f} 像素")
        
        # 将分析结果添加到保存的文本文件中
        matrix_path = os.path.join(result_dir, f"matrix_{i+1}_to_ref.txt")
        np.savetxt(matrix_path, H, fmt='%.6f')
        
        # 保存变换分析结果到单独的文件
        analysis_path = os.path.join(result_dir, f"transform_analysis_{i+1}_to_ref.txt")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            f.write(f"{image_names[i]}到{image_names[reference_idx]}的变换分析:\n")
            f.write(f"旋转角度: {rotation_angle:.2f}°\n")
            f.write(f"X方向平移: {tx:.2f} 像素\n")
            f.write(f"Y方向平移: {ty:.2f} 像素\n")
            f.write(f"总平移距离: {total_translation:.2f} 像素\n")
    
    print(f"\n所有结果已保存到: {result_dir}")

    # 保存旋转矩阵配置文件（JSON格式）
    config_path = r"D:\zycgit\ZDevelop_Confocal\xxp_ui\Algorithm\Regis_Fusion\homography_config.json"
    homography_config = {
        "H1_to_ref": homographies[0].tolist(),  # 焦后到焦面
        "H3_to_ref": homographies[2].tolist(),  # 焦前到焦面
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": "焦后和焦前到焦面的变换矩阵"
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(homography_config, f, indent=4, ensure_ascii=False)

    print(f"旋转矩阵配置已保存到: {config_path}")
    print("矩阵将自动被 Regis_Fusion.py 和 image_processor.py 使用")

    # 创建配准后的彩色合成图像
    composite_start_time = time.time()
    ref_img = images[reference_idx][0].copy()
    height, width = ref_img.shape[:2]
    
    # 并行处理图像变换
    def warp_for_channel(args):
        img, H, shape = args
        return warp_image(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), H, shape)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        warped_images = list(executor.map(
            warp_for_channel, 
            [(images[0][0], homographies[0], ref_img.shape), 
             (images[1][0], np.eye(3), ref_img.shape),
             (images[2][0], homographies[2], ref_img.shape)]
        ))
    
    # 将三张图像分别放入RGB通道
    r_channel = warped_images[0]
    g_channel = warped_images[1]
    b_channel = warped_images[2]
    
    # 合成彩色图像
    color_img = cv2.merge([b_channel, g_channel, r_channel])
    
    # 保存彩色合成图像
    color_path = os.path.join(result_dir, "color_composite.jpg")
    cv2.imwrite(color_path, color_img)
    
    composite_end_time = time.time()
    print(f"彩色合成图像生成耗时: {composite_end_time - composite_start_time:.4f} 秒")
    
    # 移除可视化代码，直接计算总执行时间
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    # 打印时间统计信息
    print("\n时间统计:")
    print(f"总执行时间: {total_time:.4f} 秒")
    print(f"图像加载时间: {load_end_time - load_start_time:.4f} 秒 ({(load_end_time - load_start_time) / total_time * 100:.2f}%)")
    print(f"特征点检测时间: {feature_end_time - feature_start_time:.4f} 秒 ({(feature_end_time - feature_start_time) / total_time * 100:.2f}%)")
    print(f"特征匹配和变换计算时间: {matching_end_time - matching_start_time:.4f} 秒 ({(matching_end_time - matching_start_time) / total_time * 100:.2f}%)")
    print(f"彩色合成图像生成时间: {composite_end_time - composite_start_time:.4f} 秒 ({(composite_end_time - composite_start_time) / total_time * 100:.2f}%)")
    
    # 保存性能报告到文本文件
    performance_path = os.path.join(result_dir, "performance_report.txt")
    with open(performance_path, 'w', encoding='utf-8') as f:
        f.write(f"GPU加速状态: {'启用' if use_gpu else '未启用'}\n")
        f.write(f"总执行时间: {total_time:.4f} 秒\n")
        f.write(f"图像加载时间: {load_end_time - load_start_time:.4f} 秒 ({(load_end_time - load_start_time) / total_time * 100:.2f}%)\n")
        f.write(f"特征点检测时间: {feature_end_time - feature_start_time:.4f} 秒 ({(feature_end_time - feature_start_time) / total_time * 100:.2f}%)\n")
        f.write(f"特征匹配和变换计算时间: {matching_end_time - matching_start_time:.4f} 秒 ({(matching_end_time - matching_start_time) / total_time * 100:.2f}%)\n")
        f.write(f"彩色合成图像生成时间: {composite_end_time - composite_start_time:.4f} 秒 ({(composite_end_time - composite_start_time) / total_time * 100:.2f}%)\n")
        f.write(f"\n特征点数量: {[len(kp) for kp in keypoints_list]}\n")
    
    print(f"性能报告已保存到: {performance_path}")
    print("配准完成，所有结果已保存")

if __name__ == "__main__":
    main()