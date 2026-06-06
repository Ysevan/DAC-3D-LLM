import cv2
import numpy as np
import os
import time
import concurrent.futures
from pathlib import Path
try:
    from .homography_loader import load_homography_matrices, get_homographies_list  # 导入配置加载工具
except ImportError:
    from homography_loader import load_homography_matrices, get_homographies_list  # 兼容直接运行本文件

# 检查CUDA是否可用
use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
print(f"GPU加速状态: {'启用' if use_gpu else '未启用'}")

# 创建结果目录
result_dir = str(Path(__file__).resolve().parents[1] / "result")
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
    """
    将图像按单应矩阵 H 配准到参考图大小。

    说明：
    - target_shape 可能是灰度图 shape=(H, W)，也可能是彩色图 shape=(H, W, C)
    - 这里统一只取前两个维度，避免彩色图传入时报 too many values to unpack
    - 保留 BORDER_REPLICATE，不产生黑边，保持当前版本的图像处理方式
    """
    use_gpu = False

    h_target, w_target = target_shape[:2]
    dsize = (w_target, h_target)

    if use_gpu:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        warped = cv2.cuda.warpPerspective(
            gpu_img, H, dsize,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        ).download()
    else:
        warped = cv2.warpPerspective(
            img, H, dsize,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )

    return warped

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
# 已替换为：几何分区融合（直边 L1 + 圆心 L2 + 三带状清晰区）
###########################################

# 几何融合参数
# FEATHER_PIX 越大，三块区域边界过渡越平滑；越小，区域切换越硬。
FEATHER_PIX = 30


def adjust_brightness_contrast(img, alpha, beta):
    """调整图像的亮度和对比度。保留原函数名，避免外部调用受影响。"""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


# ---------------- 几何检测：前景、圆心、直边 ----------------

def segment_foreground(gray):
    """Otsu 二值 + 形态学清理，返回前景 mask 和最大轮廓。"""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, m = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 保证前景占比不要反过来；如果白色区域过大，说明阈值方向可能反了
    if (m > 0).mean() > 0.5:
        m = cv2.bitwise_not(m)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return m, None

    cnt = max(cnts, key=cv2.contourArea)
    out = np.zeros_like(m)
    cv2.drawContours(out, [cnt], -1, 255, thickness=cv2.FILLED)
    return out, cnt


def _ransac_fit_circle(points, n_iters=300, inlier_tol=10.0, rng=None):
    """RANSAC 拟合圆。返回 (cx, cy, r, inlier_mask)。"""
    if rng is None:
        rng = np.random.default_rng(0)

    n = len(points)
    if n < 50:
        return None

    best = (-1, None)
    for _ in range(n_iters):
        idx = rng.choice(n, 3, replace=False)
        p1, p2, p3 = points[idx]

        ax, ay = p1
        bx, by = p2
        cx_, cy_ = p3
        d = 2 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
        if abs(d) < 1e-6:
            continue

        ux = ((ax * ax + ay * ay) * (by - cy_) +
              (bx * bx + by * by) * (cy_ - ay) +
              (cx_ * cx_ + cy_ * cy_) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx_ - bx) +
              (bx * bx + by * by) * (ax - cx_) +
              (cx_ * cx_ + cy_ * cy_) * (bx - ax)) / d

        r = np.hypot(ax - ux, ay - uy)
        if not (50 < r < 4000):
            continue

        dist = np.abs(np.hypot(points[:, 0] - ux, points[:, 1] - uy) - r)
        inliers = dist < inlier_tol
        cnt = int(inliers.sum())
        if cnt > best[0]:
            best = (cnt, (ux, uy, r, inliers))

    if best[1] is None:
        return None

    # 用所有内点最小二乘精修
    cx, cy, r, inliers = best[1]
    pts_in = points[inliers]
    A_mat = np.column_stack([2 * pts_in[:, 0], 2 * pts_in[:, 1], np.ones(len(pts_in))])
    b_vec = pts_in[:, 0] ** 2 + pts_in[:, 1] ** 2
    sol, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

    cx2, cy2 = sol[0], sol[1]
    r2 = np.sqrt(sol[2] + cx2 * cx2 + cy2 * cy2)
    dist = np.abs(np.hypot(points[:, 0] - cx2, points[:, 1] - cy2) - r2)
    inliers2 = dist < inlier_tol

    return float(cx2), float(cy2), float(r2), inliers2


def fit_circle_center(contour):
    """
    对楔形片轮廓拟合外圆。
    用 RANSAC 区分圆弧段和直边段，圆弧段拟合得到的圆心才作为真实圆心。
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    res = _ransac_fit_circle(pts, n_iters=300, inlier_tol=10.0)
    if res is None:
        (cx, cy), r = cv2.minEnclosingCircle(contour)
        return float(cx), float(cy), float(r), np.ones(len(pts), dtype=bool)

    cx, cy, r, inliers = res
    return cx, cy, r, inliers


def _ransac_filter_line(points, inlier_tol=5.0, n_iters=200, rng=None):
    """对一组 2D 点做直线 RANSAC，返回内点。"""
    if rng is None:
        rng = np.random.default_rng(1)

    n = len(points)
    if n < 5:
        return points

    best_inliers = None
    best_cnt = -1
    for _ in range(n_iters):
        idx = rng.choice(n, 2, replace=False)
        p1, p2 = points[idx]
        d = p2 - p1
        L = np.hypot(d[0], d[1])
        if L < 1e-6:
            continue

        nx, ny = -d[1] / L, d[0] / L
        dist = np.abs((points[:, 0] - p1[0]) * nx + (points[:, 1] - p1[1]) * ny)
        inliers = dist < inlier_tol
        cnt = int(inliers.sum())
        if cnt > best_cnt:
            best_cnt = cnt
            best_inliers = inliers

    if best_inliers is None or best_cnt < 5:
        return points
    return points[best_inliers]


def fit_straight_edge(contour, arc_mask=None):
    """
    用非圆弧的轮廓点拟合楔形直边 L1。
    返回 (angle, midpoint, length)。
    """
    pts = contour.reshape(-1, 2).astype(np.float32)
    if arc_mask is not None and (~arc_mask).sum() > 20:
        edge_pts = pts[~arc_mask]
    else:
        edge_pts = pts

    edge_pts = _ransac_filter_line(edge_pts, inlier_tol=5.0, n_iters=200)
    if edge_pts is None or len(edge_pts) < 5:
        return None

    vx, vy, x0, y0 = cv2.fitLine(edge_pts, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
    angle = np.arctan2(vy, vx)
    if angle >= np.pi / 2:
        angle -= np.pi
    elif angle < -np.pi / 2:
        angle += np.pi

    rel = edge_pts - np.array([x0, y0], dtype=np.float32)
    t = rel[:, 0] * vx + rel[:, 1] * vy
    t_min, t_max = float(t.min()), float(t.max())
    length = t_max - t_min
    mid_t = 0.5 * (t_min + t_max)
    mid = (float(x0 + mid_t * vx), float(y0 + mid_t * vy))

    return float(angle), mid, length


def detect_geometry(ref_gray):
    """在焦面图像上检测几何信息：直边方向、直边点、圆心、半径。"""
    h, w = ref_gray.shape
    fg, cnt = segment_foreground(ref_gray)

    if cnt is None:
        return {
            "angle": 0.0,
            "line_pt": (w / 2.0, h / 2.0),
            "center": (w / 2.0, h / 2.0),
            "radius": min(w, h) / 2.0,
        }

    cx, cy, R, arc_mask = fit_circle_center(cnt)
    edge = fit_straight_edge(cnt, arc_mask=arc_mask)
    if edge is None:
        return {"angle": 0.0, "line_pt": (cx, cy), "center": (cx, cy), "radius": R}

    angle, mid, _ = edge
    return {"angle": angle, "line_pt": mid, "center": (cx, cy), "radius": R}


# ---------------- 区域权重：L1/L2 三带状清晰区 ----------------

def build_region_weights(shape, geom, valid_mask=None):
    """
    根据几何关系构造三张权重图。

    返回顺序：[w_front, w_focus, w_back]，对应 [焦前, 焦面, 焦后]。

    本版本的关键点：
      1) 圆心、L1 直边、L2 中心线仍然从焦面图的工件几何中检测；
      2) 但三条清晰区不是只限制在工件内部，而是沿 L1/L2 方向拉长，布满整张图；
      3) 这样工件边缘、工件外背景也会被同一套带状区域覆盖，不会出现边缘缺一圈的问题。
    """
    angle = geom["angle"]
    L1_pt = geom["line_pt"]
    center = geom["center"]
    R = float(geom["radius"])

    ux, uy = np.cos(angle), np.sin(angle)
    nx, ny = -uy, ux

    # 让 d_from_L1 > 0 表示从 L1 朝圆心方向
    side = (center[0] - L1_pt[0]) * nx + (center[1] - L1_pt[1]) * ny
    if side < 0:
        nx, ny = -nx, -ny

    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)

    # 到 L1 的有符号距离：用于直边附近那条带
    d_from_L1 = (xs - L1_pt[0]) * nx + (ys - L1_pt[1]) * ny

    # 到 L2 的有符号距离：L2 是过圆心且平行 L1 的直线
    d_from_L2 = (xs - center[0]) * nx + (ys - center[1]) * ny

    # -------------------------------
    # 三条带宽度控制
    # -------------------------------
    # 直边侧带宽：从 L1 朝圆心方向，宽度约 R/2
    band_front = R / 2.0

    # 焦面中心带宽：以 L2 为中心，左右各 half_focus
    half_focus = 1.3 * R / 3.0

    # 保险外扩，避免工件边缘/区域交界处少一丝
    # 如果仍然觉得边缘有缺失，可以改成 30 或 40
    expand_pix = 30.0
    band_front += expand_pix
    half_focus += expand_pix

    # -------------------------------
    # 生成铺满整张图的三条带
    # -------------------------------
    # 注意：这里没有使用工件前景 mask 裁剪，所以条带会贯穿整张图。
    mask_front = ((d_from_L1 > 0) & (d_from_L1 < band_front)).astype(np.float32)

    mask_focus = (np.abs(d_from_L2) < half_focus).astype(np.float32)
    # 焦面带不要覆盖直边侧带
    mask_focus = np.clip(mask_focus - mask_front, 0, 1)

    # 剩余整张图区域全部给第三个焦面
    mask_back = (1.0 - mask_front - mask_focus).clip(0, 1)

    masks = [mask_front, mask_focus, mask_back]

    # 边界羽化，避免三条带之间出现硬切换
    k = FEATHER_PIX | 1
    feathered = [cv2.GaussianBlur(m, (k, k), 0) for m in masks]

    # valid_mask 只用于极端情况下的真正无效区域。
    # 当前 warp_image 使用 BORDER_REPLICATE，一般整张图都是有效的。
    # 如果你希望完全铺满全图，可以保持 valid_mask=None 或全 True。
    if valid_mask is not None:
        for i in range(len(feathered)):
            feathered[i][~valid_mask] = 0

    total = feathered[0] + feathered[1] + feathered[2]
    total[total < 1e-6] = 1.0

    return [f / total for f in feathered]

def fuse_three_geometric(images):
    """
    几何分区融合。

    参数:
        images: [焦后, 焦面, 焦前]，且都已经配准到焦面坐标系。

    返回:
        fused: 融合图像

    注意：
        这里保持整个工程的原始顺序 [焦后, 焦面, 焦前]，不再在 fusion_process() 里调换顺序。
        build_region_weights() 输出的是 [w_front, w_focus, w_back]，所以这里会重新映射成
        [w_back, w_focus, w_front] 后再和 images 对应相乘。
    """
    if len(images) != 3:
        raise ValueError(f"几何分区融合需要 3 张图像，当前输入数量: {len(images)}")

    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) if im.ndim == 3 else im for im in images]

    # 当前 images 顺序：
    # images[0] = 焦后
    # images[1] = 焦面 / 参考图
    # images[2] = 焦前
    ref_idx = 1

    # 亮度标准化：以焦面为参考
    m_ref, s_ref = cv2.meanStdDev(grays[ref_idx])
    m_ref, s_ref = m_ref[0][0], s_ref[0][0]

    adjusted = []
    for i, img in enumerate(images):
        if i == ref_idx:
            adjusted.append(img)
        else:
            mc, sc = cv2.meanStdDev(grays[i])
            alpha = s_ref / (sc[0][0] + 1e-6)
            beta = m_ref - mc[0][0] * alpha
            adjusted.append(cv2.convertScaleAbs(img, alpha=alpha, beta=beta))

    # 几何只在参考焦面图上检测，避免三张图分别检测造成区域不一致
    geom = detect_geometry(grays[ref_idx])

    # 强制使用全图有效 mask，让三条带完整铺满整张图。
    # 当前 warp_image 使用 BORDER_REPLICATE，不会产生黑色无效边。
    valid = np.ones_like(grays[ref_idx], dtype=bool)

    # build_region_weights 的输出顺序是 [焦前权重, 焦面权重, 焦后权重]
    w_front, w_focus, w_back = build_region_weights(grays[ref_idx].shape, geom, valid)

    # images 的顺序是 [焦后, 焦面, 焦前]
    # 所以权重必须映射成 [焦后权重, 焦面权重, 焦前权重]
    weights = [w_back, w_focus, w_front]

    if adjusted[0].ndim == 3:
        ws = [w[..., None] for w in weights]
    else:
        ws = weights

    fused = sum(im.astype(np.float32) * w for im, w in zip(adjusted, ws))
    fused = np.clip(fused, 0, 255).astype(np.uint8)

    return fused

def fusion_process(warped_images, use_downscale=True, downscale_factor=0.5):
    """
    替换后的融合入口函数。

    warped_images 保持原文件顺序：
        [焦后, 焦面, 焦前]

    不再在这里调换顺序，避免焦前/焦后语义混乱。
    use_downscale/downscale_factor 参数保留，是为了兼容原来的函数调用；
    几何分区融合不再使用梯度降采样权重。
    """
    if len(warped_images) != 3:
        raise ValueError(f"几何分区融合需要 3 张图像，当前输入数量: {len(warped_images)}")

    # warped_images[0] = 焦后
    # warped_images[1] = 焦面 / 参考图
    # warped_images[2] = 焦前
    return fuse_three_geometric(warped_images)


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
        result_dir = str(Path(__file__).resolve().parents[2] / "results" / "fusion_result")
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
    image_dir = r"F:\福特科\xxp_ui\Algorithm\demo_img\demo"
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



