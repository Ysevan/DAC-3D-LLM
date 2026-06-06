# -*- coding: utf-8 -*-
import queue
import torch
import random
import threading
import os
import time
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
from collections import Counter
import csv
import tempfile
import re
from datetime import datetime
import sys
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

# 确保下面的路径是正确的
sys.path.append(r"d:\zycgit\ZDevelop_Confocal\xxp_ui\Algorithm\Regis_Fusion")
from Algorithm.Regis_Fusion.Regis_Fusion import fusion_with_known_matrices
from Algorithm.Regis_Fusion.homography_loader import load_homography_matrices

# ===========================
# 新增：性能分析工具（来自第一个文件）
# ===========================
# 全局性能统计字典
performance_stats = {}

# 全局配置
ENABLE_CONTRAST_CALC = True
ENABLE_SCRATCH_ENHANCE = True
SCRATCH_CLAHE_CLIP = 0
SCRATCH_TOPHAT_LENGTH = 21
SCRATCH_BLEND_ALPHA = 0
SCRATCH_MULTI_SCALE = True
SCRATCH_CLOSING_LENGTH = 25

# GPU配置检测
def check_gpu_available():
    """检测GPU是否可用"""
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return True, gpu_name
    except:
        pass
    return False, None

def get_gpu_usage():
    """获取GPU使用情况"""
    try:
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB
            memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            utilization = (memory_allocated / memory_total) * 100
            return {
                'allocated': memory_allocated,
                'reserved': memory_reserved,
                'total': memory_total,
                'utilization': utilization
            }
    except:
        pass
    return None

GPU_AVAILABLE, GPU_NAME = check_gpu_available()

def timer_decorator(func_name=None):
    """时间统计装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal func_name
            if func_name is None:
                func_name = func.__name__
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed = end_time - start_time
            # 记录性能统计
            if func_name not in performance_stats:
                performance_stats[func_name] = {'count': 0, 'total_time': 0, 'times': []}
            performance_stats[func_name]['count'] += 1
            performance_stats[func_name]['total_time'] += elapsed
            performance_stats[func_name]['times'].append(elapsed)
            return result
        return wrapper
    return decorator

def print_performance_stats():
    """打印性能统计信息"""
    print("\n" + "="*80)
    print("性能分析报告".center(80))
    print("="*80)
    print(f"{'函数名':<30} {'调用次数':>10} {'总耗时(秒)':>15} {'平均耗时(秒)':>15}")
    print("-"*80)
    # 按总耗时排序
    sorted_stats = sorted(performance_stats.items(),
                         key=lambda x: x[1]['total_time'],
                         reverse=True)
    for func_name, stats in sorted_stats:
        avg_time = stats['total_time'] / stats['count']
        print(f"{func_name:<30} {stats['count']:>10} {stats['total_time']:>15.3f} {avg_time:>15.3f}")
    print("="*80)


PIXEL_SIZE_UM = 1.375
PHYSICAL_DIAMETER_UM = 2700.0
PHYSICAL_RADIUS_PX = PHYSICAL_DIAMETER_UM / PIXEL_SIZE_UM / 2.0
DRAW_PHYSICAL_CIRCLE = False


def fit_circle_lstsq(points):
    """
    最小二乘圆拟合：
    (x - cx)^2 + (y - cy)^2 = r^2
    """
    points = np.asarray(points, dtype=np.float64)

    x = points[:, 0]
    y = points[:, 1]

    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x ** 2 + y ** 2

    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    cx, cy, c = sol
    r = np.sqrt(c + cx ** 2 + cy ** 2)

    return float(cx), float(cy), float(r)


def ransac_circle(
    points,
    iterations=3000,
    threshold=10,
    seed=42,
    expected_radius=None,
    radius_tol=55,
    expected_center=None,
    center_tol=90,
):
    """
    受约束 RANSAC 圆拟合。

    改进点：
    1. 半径必须接近楔形片理论半径，避免拟到工装盘；
    2. 圆心不能离图像中心太远，避免拟到偏心工装圆；
    3. 打分时不仅看内点数量，也惩罚半径偏差和圆心偏差。
    """
    points = np.asarray(points, dtype=np.float64)

    if len(points) < 30:
        raise RuntimeError(f"有效边界点太少，当前只有 {len(points)} 个点")

    rng = np.random.default_rng(seed)

    best_inliers = None
    best_score = -1e18
    best_circle = None

    n = len(points)

    for _ in range(iterations):
        idx = rng.choice(n, 3, replace=False)
        sample = points[idx]

        model_path_for_loader = model_path
        try:
            model_path_for_loader = os.path.relpath(model_path, PROJECT_ROOT)
        except Exception:
            model_path_for_loader = model_path
        try:
            cx, cy, r = fit_circle_lstsq(sample)
        except Exception:
            continue

        if not np.isfinite(cx) or not np.isfinite(cy) or not np.isfinite(r):
            continue

        if r <= 0:
            continue

        radius_penalty = 0.0
        if expected_radius is not None:
            radius_err = abs(r - expected_radius)
            if radius_err > radius_tol:
                continue
            radius_penalty = radius_err

        center_penalty = 0.0
        if expected_center is not None:
            ex, ey = expected_center
            center_err = np.sqrt((cx - ex) ** 2 + (cy - ey) ** 2)
            if center_err > center_tol:
                continue
            center_penalty = center_err

        dist = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
        residual = np.abs(dist - r)

        inliers = residual < threshold
        inlier_count = int(inliers.sum())

        score = inlier_count - 0.6 * radius_penalty - 0.25 * center_penalty

        if score > best_score:
            best_score = score
            best_inliers = inliers
            best_circle = (cx, cy, r)

    if best_inliers is None or best_circle is None:
        raise RuntimeError("受约束 RANSAC 拟合失败，没有找到稳定楔形片外径圆")

    inlier_points = points[best_inliers]
    cx, cy, r = fit_circle_lstsq(inlier_points)

    if expected_radius is not None:
        if abs(r - expected_radius) > radius_tol:
            raise RuntimeError(
                f"最终拟合半径异常: r={r:.1f}, expected={expected_radius:.1f}"
            )

    if expected_center is not None:
        ex, ey = expected_center
        center_err = np.sqrt((cx - ex) ** 2 + (cy - ey) ** 2)
        if center_err > center_tol:
            raise RuntimeError(
                f"最终拟合圆心异常: center=({cx:.1f},{cy:.1f}), "
                f"expected=({ex:.1f},{ey:.1f})"
            )

    return cx, cy, r, inlier_points

def extract_outer_diameter_points(
    gray,
    init_center=None,
    num_angles=1440,
    physical_radius_px=PHYSICAL_RADIUS_PX,
    search_margin=75,
    outer_radius_window=60,
    min_inner_dark=120,
    min_rise_contrast=16,
    min_fall_drop=8,
):
    """
    提取楔形片外径边界点。

    改进点：
    1. 搜索半径围绕 2.7 mm 理论外径收紧；
    2. 外径点必须靠近理论半径；
    3. 不再盲目选择最强负梯度，而是选择“下降明显 + 半径接近理论外径”的点；
    4. 降低工装盘强反光边缘被选中的概率。
    """

    h, w = gray.shape[:2]

    if init_center is None:
        cx0, cy0 = w / 2.0, h / 2.0
    else:
        cx0, cy0 = init_center

    # 平滑，减少麻点、划痕和局部噪声对径向曲线的干扰
    img = cv2.GaussianBlur(gray, (0, 0), 2.0).astype(np.float32)

    # 外径理论半径约 981.8 px，只在附近找，避免跑到工装盘外圈
    r_min = int(max(0, physical_radius_px - search_margin))
    r_max = int(min(min(h, w) * 0.58, physical_radius_px + search_margin))

    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    outer_points = []
    inner_points = []

    for theta in angles:
        rs = np.arange(r_min, r_max, dtype=np.float32)

        xs = cx0 + rs * np.cos(theta)
        ys = cy0 + rs * np.sin(theta)

        valid = (xs >= 1) & (xs < w - 2) & (ys >= 1) & (ys < h - 2)

        if valid.sum() < 80:
            continue

        xs_valid = xs[valid].astype(np.float32)
        ys_valid = ys[valid].astype(np.float32)
        rs_valid = rs[valid]

        profile = cv2.remap(
            img,
            xs_valid.reshape(1, -1),
            ys_valid.reshape(1, -1),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        ).reshape(-1)

        if len(profile) < 80:
            continue

        profile_smooth = cv2.GaussianBlur(
            profile.reshape(-1, 1),
            (1, 31),
            0
        ).reshape(-1)

        grad = np.diff(profile_smooth)

        margin = 20
        if len(grad) <= 2 * margin:
            continue

        # -------------------------------------------------
        # Step 1：找内边缘：黑色内部 -> 亮边
        # -------------------------------------------------
        search_grad = grad[margin:-margin]
        inner_idx = int(np.argmax(search_grad)) + margin

        inside_start = max(0, inner_idx - 35)
        inside_end = inner_idx

        after_start = inner_idx + 1
        after_end = min(len(profile_smooth), inner_idx + 36)

        if inside_end <= inside_start or after_end <= after_start:
            continue

        inside_mean = float(profile_smooth[inside_start:inside_end].mean())
        after_mean = float(profile_smooth[after_start:after_end].mean())
        rise_contrast = after_mean - inside_mean

        # 内部必须是暗区，并且后面要明显变亮
        if inside_mean > min_inner_dark:
            continue

        if rise_contrast < min_rise_contrast:
            continue

        r_inner = rs_valid[inner_idx]
        x_inner = cx0 + r_inner * np.cos(theta)
        y_inner = cy0 + r_inner * np.sin(theta)
        inner_points.append([x_inner, y_inner])

        # -------------------------------------------------
        # Step 2：从内边缘往外找亮带峰值
        # -------------------------------------------------
        peak_start = inner_idx + 5
        peak_end = min(len(profile_smooth) - 10, inner_idx + 80)

        if peak_end <= peak_start:
            continue

        peak_local = int(np.argmax(profile_smooth[peak_start:peak_end]))
        peak_idx = peak_start + peak_local

        # -------------------------------------------------
        # Step 3：从峰值后找外径下降边
        # 注意：这里不是直接取最强负梯度，
        # 而是取“负梯度明显 + 半径接近理论外径”的点。
        # -------------------------------------------------
        fall_start = peak_idx + 4
        fall_end = min(len(grad) - 5, peak_idx + 70)

        if fall_end <= fall_start:
            continue

        candidate_idxs = np.arange(fall_start, fall_end, dtype=np.int32)

        # 候选点对应的半径
        candidate_rs = rs_valid[candidate_idxs]

        # 必须在理论外径附近，避免跑到工装盘
        radius_err = np.abs(candidate_rs - physical_radius_px)
        radius_valid = radius_err <= outer_radius_window

        if not np.any(radius_valid):
            continue

        candidate_idxs = candidate_idxs[radius_valid]
        radius_err = radius_err[radius_valid]

        # 负梯度越大，越像亮边外侧
        neg_grad = -grad[candidate_idxs]

        # 至少要是下降边
        valid_drop = neg_grad > 0

        if not np.any(valid_drop):
            continue

        candidate_idxs = candidate_idxs[valid_drop]
        radius_err = radius_err[valid_drop]
        neg_grad = neg_grad[valid_drop]

        # 综合评分：下降越明显越好，离理论外径越近越好
        score = neg_grad - 0.08 * radius_err

        outer_idx = int(candidate_idxs[np.argmax(score)])

        before_outer = float(
            profile_smooth[max(0, outer_idx - 20):outer_idx].mean()
        )
        after_outer = float(
            profile_smooth[outer_idx + 1:min(len(profile_smooth), outer_idx + 31)].mean()
        )

        fall_drop = before_outer - after_outer

        if fall_drop < min_fall_drop:
            continue

        r_outer = rs_valid[outer_idx]

        # 再做一次硬约束
        if abs(r_outer - physical_radius_px) > outer_radius_window:
            continue

        x_outer = cx0 + r_outer * np.cos(theta)
        y_outer = cy0 + r_outer * np.sin(theta)

        outer_points.append([x_outer, y_outer])

    outer_points = np.asarray(outer_points, dtype=np.float32)
    inner_points = np.asarray(inner_points, dtype=np.float32)

    return outer_points, inner_points

@timer_decorator()
def detect_and_mask_circle(image):
    """
    使用楔形片外径拟圆方法检测圆形并输出掩码。
    完全兼容原接口：return masked_img, circle_info

    注意：
    - circle_info['radius'] 现在表示楔形片外径半径；
    - 下游区域 A/B/C/D 判定逻辑不需要改；
    - mask 仍然用外径半径 + 20，保持你原来的使用方式。
    """
    if image is None:
        return None, None

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img = image.copy()
    else:
        gray = image.copy()
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    h, w = gray.shape[:2]

    if gray.dtype != np.uint8:
        gray_8u = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        gray_8u = gray

    init_center = (w / 2.0, h / 2.0)

    try:
        outer_points, inner_points = extract_outer_diameter_points(
            gray_8u,
            init_center=init_center,
            num_angles=1440,
            physical_radius_px=PHYSICAL_RADIUS_PX,
            search_margin=75,
            outer_radius_window=60,
            min_inner_dark=120,
            min_rise_contrast=16,
            min_fall_drop=8,
        )

        if len(outer_points) < 30:
            return None, None

        cx, cy, r_outer_fit, inlier_points = ransac_circle(
            outer_points,
            iterations=4000,
            threshold=10,
            seed=42,
            expected_radius=PHYSICAL_RADIUS_PX,
            radius_tol=55,
            expected_center=init_center,
            center_tol=90,
        )

    except Exception as e:
        print(f"外径拟圆失败: {e}")
        return None, None

    # 保持你原来的逻辑：拟合半径外扩 20 px 作为 mask 半径
    R_extended = r_outer_fit + 50

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask,(int(round(cx)), int(round(cy))),int(round(R_extended)),255,-1)

    masked_img = cv2.bitwise_and(img, img, mask=mask)

    circle_info = {
        'center': (cx, cy),
        'radius': r_outer_fit,
        'extended_radius': R_extended
    }

    return masked_img, circle_info

# ===========================
# 弱划痕预处理增强（不变）
# ===========================
def enhance_weak_scratches(img, clahe_clip=3.0, tophat_length=21, blend_alpha=0.6,
                            multi_scale=True, closing_length=25):
    """
    弱划痕预处理增强（改进版）
    """
    is_color = len(img.shape) == 3
    if is_color:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    scratch_map = np.zeros_like(gray, dtype=np.float32)
    angles = [0, 30, 60, 90, 120, 150]
    if multi_scale:
        scales = [tophat_length, tophat_length * 2, tophat_length * 3]
    else:
        scales = [tophat_length]

    for scale in scales:
        for angle in angles:
            kernel = np.zeros((scale, scale), dtype=np.uint8)
            cx = scale // 2
            for i in range(scale):
                rad = np.radians(angle)
                x = int(cx + (i - cx) * np.cos(rad))
                y = int(cx + (i - cx) * np.sin(rad))
                if 0 <= x < scale and 0 <= y < scale:
                    kernel[y, x] = 1
            if kernel.sum() == 0:
                kernel[cx, :] = 1
            black_hat = cv2.morphologyEx(gray_clahe, cv2.MORPH_BLACKHAT, kernel)
            white_hat = cv2.morphologyEx(gray_clahe, cv2.MORPH_TOPHAT, kernel)
            scratch_map = np.maximum(scratch_map, black_hat.astype(np.float32))
            scratch_map = np.maximum(scratch_map, white_hat.astype(np.float32))

    if closing_length > 0:
        closed_map = np.zeros_like(scratch_map)
        scratch_uint8 = np.clip(scratch_map, 0, 255).astype(np.uint8)
        for angle in angles:
            close_kernel = np.zeros((closing_length, closing_length), dtype=np.uint8)
            cx = closing_length // 2
            for i in range(closing_length):
                rad = np.radians(angle)
                x = int(cx + (i - cx) * np.cos(rad))
                y = int(cx + (i - cx) * np.sin(rad))
                if 0 <= x < closing_length and 0 <= y < closing_length:
                    close_kernel[y, x] = 1
            if close_kernel.sum() == 0:
                close_kernel[cx, :] = 1
            closed = cv2.morphologyEx(scratch_uint8, cv2.MORPH_CLOSE, close_kernel)
            closed_map = np.maximum(closed_map, closed.astype(np.float32))
        scratch_map = closed_map

    if is_color:
        enhanced = img.astype(np.float32)
        for c in range(3):
            enhanced[:, :, c] = img[:, :, c].astype(np.float32) + scratch_map * blend_alpha
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    else:
        enhanced = np.clip(
            img.astype(np.float32) + scratch_map * blend_alpha, 0, 255
        ).astype(np.uint8)
    return enhanced

# ===========================
# 对比度指标计算（不变）
# ===========================
@timer_decorator()
def calculate_contrast_metrics(original_img, defect_position, original_gray_cache=None):
    x1, y1, x2, y2 = [int(v) for v in defect_position]
    img_height, img_width = original_img.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_width - 1, x2)
    y2 = min(img_height - 1, y2)
    defect_region = original_img[y1:y2 + 1, x1:x2 + 1]

    if defect_region.size == 0 or defect_region.shape[0] < 3 or defect_region.shape[1] < 3:
        return {
            'weber_contrast': 0.0, 'michelson_contrast': 0.0, 'rms_contrast': 0.0,
            'cnr': 0.0, 'dsnr': 0.0, 'contrast_size_product': 0.0, 'local_contrast': 0.0,
            'snr': 0.0, 'edge_strength': 0.0, 'defect_mean': 0.0, 'background_mean': 0.0,
            'background_std': 0.0, 'defect_std': 0.0, 'defect_size': (0, 0),
            'contrast_level': 'unknown', 'dsnr_level': 'unknown'
        }

    if original_gray_cache is not None:
        original_gray = original_gray_cache
    elif len(original_img.shape) == 3:
        original_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original_img

    if len(defect_region.shape) == 3:
        defect_gray = cv2.cvtColor(defect_region, cv2.COLOR_BGR2GRAY)
    else:
        defect_gray = defect_region

    defect_gray_float = defect_gray.astype(np.float64)
    original_gray_float = original_gray.astype(np.float64)

    defect_mean = float(np.mean(defect_gray_float))
    defect_std = float(np.std(defect_gray_float))
    defect_max = float(np.max(defect_gray_float))
    defect_min = float(np.min(defect_gray_float))
    defect_size = (defect_gray.shape[1], defect_gray.shape[0])
    defect_area = defect_gray.shape[0] * defect_gray.shape[1]

    expand_factor = 2
    bg_x1 = max(0, int(x1 - (x2 - x1) * expand_factor))
    bg_y1 = max(0, int(y1 - (y2 - y1) * expand_factor))
    bg_x2 = min(img_width - 1, int(x2 + (x2 - x1) * expand_factor))
    bg_y2 = min(img_height - 1, int(y2 + (y2 - y1) * expand_factor))

    bg_region = original_gray_float[bg_y1:bg_y2 + 1, bg_x1:bg_x2 + 1]
    mask = np.ones(bg_region.shape, dtype=bool)

    rel_x1 = x1 - bg_x1
    rel_y1 = y1 - bg_y1
    rel_x2 = x2 - bg_x1
    rel_y2 = y2 - bg_y1

    if rel_y1 >= 0 and rel_y2 < mask.shape[0] and rel_x1 >= 0 and rel_x2 < mask.shape[1]:
        mask[rel_y1:rel_y2 + 1, rel_x1:rel_x2 + 1] = False

    background_pixels = bg_region[mask]
    if background_pixels.size == 0:
        background_mean = defect_mean
        background_std = defect_std
    else:
        background_mean = float(np.mean(background_pixels))
        background_std = float(np.std(background_pixels))

    weber_contrast = abs(defect_mean - background_mean) / (background_mean + 1e-10)
    michelson_contrast = (defect_max - defect_min) / (defect_max + defect_min + 1e-10)
    rms_contrast = defect_std / (defect_mean + 1e-10)
    cnr = abs(defect_mean - background_mean) / (background_std + 1e-10)
    dsnr = abs(defect_mean - background_mean) / (background_std + 1e-10)
    contrast_size_product = weber_contrast * defect_area
    local_contrast = abs(defect_mean - background_mean) / (defect_mean + background_mean + 1e-10)
    snr = defect_mean / (background_std + 1e-10)

    edge_strength = 0.0
    if defect_gray_float.shape[0] > 1 and defect_gray_float.shape[1] > 1:
        sobelx = cv2.Sobel(defect_gray_float, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(defect_gray_float, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        edge_strength = float(np.mean(edge_magnitude))

    if weber_contrast > 0.2:
        contrast_level = '高对比度'
    elif weber_contrast >= 0.1:
        contrast_level = '中对比度'
    elif weber_contrast >= 0.03:
        contrast_level = '低对比度'
    else:
        contrast_level = '极低对比度'

    if dsnr >= 10:
        dsnr_level = '高对比度（常规算法可检）'
    elif dsnr >= 6:
        dsnr_level = '中对比度（简单增强可检）'
    elif dsnr >= 3:
        dsnr_level = '低对比度（需专用算法）'
    else:
        dsnr_level = '极低对比度（需优化成像）'

    return {
        'weber_contrast': float(weber_contrast), 'michelson_contrast': float(michelson_contrast),
        'rms_contrast': float(rms_contrast), 'cnr': float(cnr), 'dsnr': float(dsnr),
        'contrast_size_product': float(contrast_size_product), 'local_contrast': float(local_contrast),
        'snr': float(snr), 'edge_strength': float(edge_strength), 'defect_mean': float(defect_mean),
        'background_mean': float(background_mean), 'background_std': float(background_std),
        'defect_std': float(defect_std), 'defect_size': defect_size, 'defect_area': defect_area,
        'contrast_level': contrast_level, 'dsnr_level': dsnr_level
    }

# ===========================
# 分区域瑕疵合格性检查（不变）
# ===========================
def check_defect_in_region(defect_type, defect_size, defect_info, all_defects, region_ratio, radius):
    width, height = defect_size
    pixel_factor = 5.5 / 4
    max_size = max(width, height)

    if 0 <= region_ratio < 0.333:
        if defect_type == 'splash':
            threshold = 10 / pixel_factor
            if max_size >= threshold:
                return False, f"区域A中发现超标麻点（尺寸{width:.1f}x{height:.1f}px，阈值{threshold:.1f}px）"
        elif defect_type == 'scratch':
            threshold = 300 / pixel_factor
            if max_size >= threshold:
                return False, f"区域A中发现超标划痕（尺寸{width:.1f}x{height:.1f}px，阈值{threshold:.1f}px）"
            scratch_count = sum(1 for d in all_defects if d['type'] == 'scratch' and d['region_ratio'] < 0.409)
            if scratch_count >= 2:
                return False, f"区域A中划痕数量超标（{scratch_count}条，允许<2条）"

    elif 0.333 <= region_ratio < 0.519:
        if defect_type == 'splash':
            min_threshold = 10 / pixel_factor
            max_threshold = 50 / pixel_factor
            if max_size < min_threshold:
                return True, ""
            elif max_size > max_threshold:
                return False, f"区域B中发现过大麻点（尺寸{width:.1f}x{height:.1f}px，最大阈值{max_threshold:.1f}px）"
            else:
                valid_splash_count = sum(1 for d in all_defects if d['type'] == 'splash' and 0.333 <= d['region_ratio'] < 0.519 and min_threshold <= max(d['size']) <= max_threshold)
                if valid_splash_count > 5:
                    return False, f"区域B中(10-50μm)麻点数量超标（{valid_splash_count}个，允许≤5个）"
        elif defect_type == 'scratch':
            threshold = 40 / pixel_factor
            if max_size >= threshold:
                return False, f"区域B中发现超标划痕（尺寸{width:.1f}x{height:.1f}px，阈值{threshold:.1f}px）"
            total_scratch_length = sum(max(d['size']) for d in all_defects if d['type'] == 'scratch' and 0.333<= d['region_ratio'] < 0.519)
            max_total = 67.5 / pixel_factor
            if total_scratch_length > max_total:
                return False, f"区域B中划痕总长度超标（{total_scratch_length:.1f}px，阈值{max_total:.1f}px）"

    elif 0.519 <= region_ratio < 0.815:
        if defect_type == 'splash':
            max_threshold = 100 / pixel_factor
            if max_size > max_threshold:
                return False, f"区域C中发现超大麻点（尺寸{width:.1f}x{height:.1f}px，最大阈值{max_threshold:.1f}px）"
        elif defect_type == 'scratch':
            threshold = 40 / pixel_factor
            if max_size >= threshold:
                return False, f"区域C中发现超标划痕（尺寸{width:.1f}x{height:.1f}px，阈值{threshold:.1f}px）"

    elif 0.815 <= region_ratio <= 1:
        if defect_type == 'chipping':
            threshold = 100 / pixel_factor
            if width >= threshold or height >= threshold:
                return False, f"区域D中发现超标崩边（尺寸{width:.1f}x{height:.1f}px，阈值{threshold:.1f}px）"
        elif defect_type in ['splash', 'scratch']:
            return True, "ignore"

    return True, ""

# ===========================
# ImageProcessor 主类（完全不变）
# ===========================
class ImageProcessor:
    def __init__(self, debug_image_queue, image_debug_queue, ui_image_queue, image_ui_queue):
        self.pos = 0
        self.Tray_id = None
        self.debug_image_queue = debug_image_queue
        self.image_debug_queue = image_debug_queue
        self.ui_image_queue = ui_image_queue
        self.image_ui_queue = image_ui_queue
        self.pending_sample_data = {}
        self.scan_part_counter = 0
        self.final_sample_position = 0
        self.model_configs = {
            "old": {
                "type": "sahi_path",
                "model_path": r"D:\zycgit\ZDevelop_Confocal\xxp_ui\deploy\weights\best.pt",
                "conf": 0.25,
            },
            "new": {
                "type": "sahi_yolo_preload",
                "model_path": r"D:\zycgit\ZDevelop_Confocal\xxp_ui\deploy\weights\best.torchscript",
                "conf": 0.25,
            }
        }
        self.active_model_key = "old"
        self.sahi_model = None
        self.model_loaded = False
        self.load_defect_model(self.active_model_key)
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.run_result_dir = os.path.join(
            r"E:\ftkpic\results",
            self.run_timestamp
        )
        self.fusion_result_dir = os.path.join(
            self.run_result_dir,
            "融合结果"
        )
        self.result_dir = os.path.join(
            self.run_result_dir,
            "检测结果"
        )
        os.makedirs(self.fusion_result_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        self.getInfoFromDebug_thread = threading.Thread(target=self.getInfoFromDebug)
        self.getInfoFromDebug_thread.start()
        self.getInfoFromUI_thread = threading.Thread(target=self.getInfoFromUI)
        self.getInfoFromUI_thread.start()

    def load_defect_model(self, model_key="old"):
        import traceback
        self.model_loaded = False
        self.sahi_model = None
        cfg = self.model_configs.get(model_key, None)
        if cfg is None:
            print(f"[ERR] 未知 model_key: {model_key} (可选: {list(self.model_configs.keys())})")
            return
        model_path = cfg["model_path"]
        conf = float(cfg.get("conf", 0.25))
        print("\n========== load_defect_model ==========")
        print("model_key:", model_key)
        print("model_path:", model_path)
        print("exists:", os.path.exists(model_path))
        print("torch:", torch.__version__)
        print("cuda_available:", torch.cuda.is_available())
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("inference_device:", device)
        if not os.path.exists(model_path):
            print(f"[ERR] 模型文件不存在: {model_path}")
            return
        try:
            if model_key == "old":
                self.sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="ultralytics",
                    model_path=model_path_for_loader,
                    confidence_threshold=conf,
                    device=device
                )
            elif model_key == "new":
                yolo_model = YOLO(model_path_for_loader, task="detect")
                try:
                    yolo_model.to(device)
                except Exception as move_error:
                    print(f"[WARN] YOLO model move to {device} failed: {move_error}")
                try:
                    self.sahi_model = AutoDetectionModel.from_pretrained(
                        model_type="ultralytics",
                        model=yolo_model,
                        confidence_threshold=conf,
                        device=device,
                    )
                except TypeError:
                    self.sahi_model = AutoDetectionModel.from_pretrained(
                        model_type="ultralytics",
                        model=yolo_model,
                        confidence_threshold=conf,
                    )
                print("YOLO internal model:", type(getattr(yolo_model, "model", None)))
                print("SAHI detection_model device:", getattr(self.sahi_model, "device", None))
            self.model_loaded = True
            self.active_model_key = model_key
            print(f"[OK] 模型加载成功，当前模式: {self.active_model_key}")
        except Exception as e:
            print(f"[ERR] 加载模型失败({model_key}): {e}")
            traceback.print_exc()
            self.model_loaded = False
            self.sahi_model = None

    @timer_decorator()
    def process_image_with_sahi(self, frame, conf_threshold=0.25, slice_height=512, slice_width=512):
        try:
            print("  >> 执行圆形拟合...")
            masked_frame, circle_info = detect_and_mask_circle(frame)
            if masked_frame is None or circle_info is None:
                print("  >> 警告: 圆形拟合失败，使用原始图像进行检测")
                masked_frame = frame
                circle_info = None
            else:
                center_x, center_y = circle_info['center']
                radius = circle_info['radius']
                print(f"  >> 圆形拟合成功 - 圆心: ({center_x:.1f}, {center_y:.1f}), 半径: {radius:.1f}px")

            if ENABLE_SCRATCH_ENHANCE:
                print("  >> 执行弱划痕增强...")
                infer_img = enhance_weak_scratches(
                    masked_frame,
                    clahe_clip=SCRATCH_CLAHE_CLIP,
                    tophat_length=SCRATCH_TOPHAT_LENGTH,
                    blend_alpha=SCRATCH_BLEND_ALPHA,
                    multi_scale=SCRATCH_MULTI_SCALE,
                    closing_length=SCRATCH_CLOSING_LENGTH
                )
            else:
                infer_img = masked_frame

            if len(infer_img.shape) == 2:
                rgb_frame = cv2.cvtColor(infer_img, cv2.COLOR_GRAY2RGB)
            else:
                rgb_frame = infer_img.copy()

            print(f"  >> 当前检测模型: {self.active_model_key}")
            print("  >> 执行 SAHI 切片检测...")
            result = get_sliced_prediction(
                image=rgb_frame,
                detection_model=self.sahi_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=0.1,
                overlap_width_ratio=0.1,
                perform_standard_pred=True,
                postprocess_type="NMS",
                postprocess_match_threshold=0.9,
                postprocess_match_metric="IOU"
            )

            annotator = Annotator(rgb_frame, line_width=2)
            detection_data = [
                (det.category.name, det.category.id, (det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy),
                 float(det.score.value))
                for det in result.object_prediction_list
            ]
            cnt = 0
            defect_details = []
            all_defects_for_check = []
            unqualified_reasons = []
            defects_to_mark = []

            original_gray_cache = None
            if ENABLE_CONTRAST_CALC:
                if len(frame.shape) == 3:
                    original_gray_cache = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    original_gray_cache = frame

            for det in detection_data:
                minx, miny, maxx, maxy = det[2]
                width = maxx - minx
                height = maxy - miny
                defect_type = det[0]
                defect_center_x = (minx + maxx) / 2
                defect_center_y = (miny + maxy) / 2
                region_ratio = -1
                if circle_info is not None:
                    cx, cy = circle_info['center']
                    r = circle_info['radius']
                    distance = np.sqrt((defect_center_x - cx)**2 + (defect_center_y - cy)**2)
                    region_ratio = distance / r
                defect_info = {
                    'type': defect_type,
                    'size': (width, height),
                    'position': (minx, miny, maxx, maxy),
                    'center': (defect_center_x, defect_center_y),
                    'region_ratio': region_ratio
                }
                all_defects_for_check.append(defect_info)

            for det in detection_data:
                cnt += 1
                minx, miny, maxx, maxy = det[2]
                confidence = det[3]
                width = maxx - minx
                height = maxy - miny
                defect_type = det[0]
                defect_center_x = (minx + maxx) / 2
                defect_center_y = (miny + maxy) / 2
                region_ratio = -1
                if circle_info is not None:
                    cx, cy = circle_info['center']
                    r = circle_info['radius']
                    distance = np.sqrt((defect_center_x - cx)**2 + (defect_center_y - cy)**2)
                    region_ratio = distance / r

                contrast_metrics = None
                if defect_type == 'splash' and ENABLE_CONTRAST_CALC and original_gray_cache is not None:
                    try:
                        contrast_metrics = calculate_contrast_metrics(
                            frame, (minx, miny, maxx, maxy), original_gray_cache
                        )
                    except Exception as e:
                        pass

                is_qualified = True
                reason = ""
                if circle_info is not None:
                    is_qualified, reason = check_defect_in_region(
                        defect_type, (width, height),
                        {'type': defect_type, 'size': (width, height), 'region_ratio': region_ratio},
                        all_defects_for_check,
                        region_ratio,
                        circle_info['radius']
                    )
                if reason == "ignore":
                    continue
                if not is_qualified:
                    unqualified_reasons.append(reason)

                defect_info = {
                    "id": cnt, "category": defect_type, "category_id": int(det[1]),
                    "confidence": float(confidence),
                    "bbox": {
                        "x1": float(minx), "y1": float(miny), "x2": float(maxx),
                        "y2": float(maxy), "width": float(width), "height": float(height)
                    },
                    "region_ratio": region_ratio,
                    "is_qualified": is_qualified,
                    "reason": reason
                }
                if contrast_metrics is not None:
                    defect_info['contrast_metrics'] = contrast_metrics
                defect_details.append(defect_info)
                defects_to_mark.append((det[2], defect_type, confidence, is_qualified, region_ratio))

            result_img = annotator.result()
            for box_info in defects_to_mark:
                box, defect_type, conf, is_qualified, region_ratio = box_info
                if not is_qualified:
                    box_color = (0, 0, 255)
                    text_color = (255, 255, 255)
                else:
                    box_color = (0, 255, 0)
                    text_color = (0, 0, 0)
                if region_ratio < 0.333:
                    region_label = "A"
                elif 0.333 < region_ratio < 0.519:
                    region_label = "B"
                elif 0.519  < region_ratio < 0.815:
                    region_label = "C"
                elif 0.815 < region_ratio < 1:
                    region_label = "D"
                else:
                    region_label = "OUT"
                label = f"{defect_type}{region_label} {conf:.2f}"
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(result_img, (x1, y1), (x2, y2), box_color, 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                thickness = 1
                (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                cv2.rectangle(result_img, (x1, y1 - th - 8), (x1 + tw + 6, y1), box_color, -1)
                cv2.putText(result_img, label, (x1 + 3, y1 - 4), font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
                cv2.putText(result_img, label, (x1 + 3, y1 - 4), font, font_scale, text_color, thickness, cv2.LINE_AA)

            if circle_info is not None:
                center_x, center_y = int(circle_info['center'][0]), int(circle_info['center'][1])
                radius = int(circle_info['radius'])
                cv2.circle(result_img, (center_x, center_y), radius, (255, 255, 255), 2)
                cv2.circle(result_img, (center_x, center_y), 5, (0, 255, 0), -1)
                regions = [0.333, 0.519, 0.815,1]
                colors = [(255, 100, 0), (0, 200, 0), (0, 100, 255), (255, 255, 0)]
                for i, ratio in enumerate(regions):
                    r = int(radius * ratio)
                    thickness = 2 if i == len(regions) - 1 else 1
                    cv2.circle(result_img, (center_x, center_y), r, colors[i], thickness)

            print(f"  >> 检测完成，共检测到 {cnt} 个瑕疵，不合格 {len(unqualified_reasons)} 个")
            if unqualified_reasons:
                for r in unqualified_reasons[:3]:
                    print(f"     - {r}")
                if len(unqualified_reasons) > 3:
                    print(f"     ... (还有{len(unqualified_reasons)-3}个)")
            return result_img, result, cnt, defect_details

        except Exception as e:
            print(f"处理图像时出错: {e}")
            import traceback
            traceback.print_exc()
            return None, None, 0, []

    def getInfoFromDebug(self):
        while True:
            try:
                data = self.debug_image_queue.get(timeout=2)
                if not data:
                    continue
                part_number = self.scan_part_counter + 1
                surface_type = "上表面" if part_number == 1 else "下表面"
                print(f"图像处理进程接收到样本的第 {part_number} 部分 ({surface_type}) 图像...")

                Image.fromarray(data['DA3562103']['frame']).save(data['DA3562103']['pic_path'])
                img_2117_rotated = Image.fromarray(data['DA3562117']['frame'])
                img_2117=np.array(img_2117_rotated)
                flip_img=cv2.flip(img_2117, 1)
                flip_img_pil = Image.fromarray(flip_img)
                flip_img_pil.save(data['DA3562117']['pic_path'])
                img_7093_flipped = Image.fromarray(data['DA3827093']['frame'])
                img_7093_flipped.save(data['DA3827093']['pic_path'])
                current_pic_paths = [data['DA3562103']['pic_path'], data['DA3562117']['pic_path'], data['DA3827093']['pic_path']]
                pos_for_filename = self.final_sample_position + 1

                current_result_pic_path = os.path.join(
                    self.result_dir,
                    f"pos{pos_for_filename}_surface{part_number}_detect.jpg"
                )

                print(f"开始对第 {part_number} 部分 ({surface_type}) 进行图像配准和融合处理...")
                fusion_start_time = time.time()
                img_back = data['DA3827093']['frame'].copy()
                img_focus = data['DA3562103']['frame'].copy()
                img_front = data['DA3562117']['frame'].copy()
                flip_img_front = cv2.flip(img_front, 1)
                fusion_image_list = [img_back, img_focus, flip_img_front]

                pos_for_filename = self.final_sample_position + 1
                pre_fusion_dir = os.path.join(self.fusion_result_dir, "pre_fusion_images")
                os.makedirs(pre_fusion_dir, exist_ok=True)
                img_back_path = os.path.join(pre_fusion_dir, f"pos{pos_for_filename}_surface{part_number}_焦后.jpg")
                cv2.imwrite(img_back_path, img_back)
                img_focus_path = os.path.join(pre_fusion_dir, f"pos{pos_for_filename}_surface{part_number}_焦面.jpg")
                cv2.imwrite(img_focus_path, img_focus)
                flip_img_front_path = os.path.join(pre_fusion_dir, f"pos{pos_for_filename}_surface{part_number}_焦前.jpg")
                cv2.imwrite(flip_img_front_path, flip_img_front)

                H1_to_ref, H3_to_ref = load_homography_matrices()
                homographies = [H1_to_ref, np.eye(3), H3_to_ref]
                fused_image, warped_images = fusion_with_known_matrices(
                    fusion_image_list,
                    homographies=homographies,
                    result_dir=self.fusion_result_dir,
                    verbose=True,
                    use_downscale=False
                )
                fusion_elapsed_time = time.time() - fusion_start_time
                print(f"图像融合完成，耗时: {fusion_elapsed_time:.2f} 秒")

                pos_for_filename = self.final_sample_position + 1
                fusion_result_filename = f"fused_image_pos{pos_for_filename}_surface{part_number}.jpg"
                fusion_result_path = os.path.join(self.fusion_result_dir, fusion_result_filename)
                if fused_image is not None:
                    cv2.imwrite(fusion_result_path, fused_image)
                    print(f"融合结果已保存为: {fusion_result_path}")

                defections = []
                res = 1
                result_img = None
                if self.model_loaded:
                    start_time = time.time()
                    result_img, _, defect_count, defect_details = self.process_image_with_sahi(
                        fused_image, conf_threshold=0.25, slice_height=512, slice_width=512
                    )
                    elapsed_time = time.time() - start_time
                    print(f"瑕疵检测耗时: {elapsed_time:.2f} 秒，检测到 {defect_count} 个瑕疵")
                    if result_img is not None:
                        cv2.imwrite(current_result_pic_path, result_img)
                    has_unqualified = any(not d.get('is_qualified', True) for d in defect_details)
                    if has_unqualified:
                        res = 0
                        for defect in defect_details:
                            if not defect.get('is_qualified', True):
                                bbox = defect["bbox"]
                                defections.append({
                                    'defection_type': defect["category"],
                                    'defection_pos': ((bbox["x1"] + bbox["x2"]) / 2, (bbox["y1"] + bbox["y2"]) / 2),
                                    'defection_size': max(bbox["width"], bbox["height"]),
                                    'reason': defect.get('reason', '')
                                })
                else:
                    print("警告: 模型未加载，使用随机结果进行模拟。")
                    res = random.randint(0, 1)

                if self.scan_part_counter == 0:
                    self.pending_sample_data = {
                        'quality': res, 'pic_path': current_pic_paths, 'defections': defections,
                        'result_pic_path': current_result_pic_path, 'result_img': result_img
                    }
                    self.scan_part_counter = 1
                    print("样本第一部分处理完成，等待第二部分...\n")
                elif self.scan_part_counter == 1:
                    print("样本第二部分处理完成，开始合并最终结果...")
                    first_part_data = self.pending_sample_data
                    final_res = 1 if first_part_data['quality'] == 1 and res == 1 else 0
                    final_defections = first_part_data['defections'] + defections
                    self.final_sample_position += 1
                    result_dir = self.result_dir
                    os.makedirs(result_dir, exist_ok=True)

                    tray_id = self.Tray_id if self.Tray_id is not None else "unknown_tray"
                    base_name = f"{tray_id}_pos{self.final_sample_position}"
                    surface1_path = os.path.join(result_dir, f"{base_name}_surface下.jpg")
                    if first_part_data['result_img'] is not None:
                        cv2.imwrite(surface1_path, first_part_data['result_img'])
                    surface2_path = os.path.join(result_dir, f"{base_name}_surface上.jpg")
                    if result_img is not None:
                        cv2.imwrite(surface2_path, result_img)

                    final_sample_data = {
                        'Tray_id': self.Tray_id, 'quality': final_res,
                        'pic_path': {'surface_1': first_part_data['pic_path'], 'surface_2': current_pic_paths},
                        'pos': self.final_sample_position,
                        'defections': final_defections,
                        'result_pic_path': {'surface_1': first_part_data['result_pic_path'],
                                            'surface_2': current_result_pic_path},
                        'result_img': [result_img , first_part_data['result_img']]
                    }
                    print(final_sample_data)
                    self.image_ui_queue.put({'type': 'Sample_detection_result', 'sample_data': final_sample_data})
                    print(f"样本 {self.final_sample_position} 的最终结果已发送到UI。最终质量: {'合格' if final_res == 1 else '不合格'}\n")
                    self.pending_sample_data = {}
                    self.scan_part_counter = 0
                    if self.final_sample_position >= 144:
                        self.final_sample_position = 0
                        self.image_ui_queue.put({'type': 'info', 'data': 'finish'})
                        print("所有样本处理完成！")
                        print_performance_stats()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"图像处理过程中发生严重错误: {e}")
                import traceback
                traceback.print_exc()
                self.pending_sample_data = {}
                self.scan_part_counter = 0

    def getInfoFromUI(self):
        while True:
            try:
                data = self.ui_image_queue.get(timeout=2)
                if data:
                    print('图像处理进程接收到UI消息：', data)
                    if data['type'] == 'info':
                        self.Tray_id = data['data']['tray_id']
                    elif data['type'] == 'reload_model':
                        self.load_defect_model()
            except queue.Empty:
                pass
