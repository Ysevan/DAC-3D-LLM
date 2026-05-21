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

PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = PROJECT_ROOT / "runtime" / "ftkpic"

# 确保下面的路径是正确的
sys.path.append(str(PROJECT_ROOT / "Algorithm" / "Regis_Fusion"))
from Algorithm.Regis_Fusion.Regis_Fusion_three2 import fusion_with_known_matrices
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


# ===========================
# 新增：瑕疵 CSV 明细与区域统计输出
# ===========================
def get_region_label(region_ratio):
    """根据 region_ratio 返回 A/B/C/D/OUT 区域标签。"""
    try:
        r = float(region_ratio)
    except Exception:
        return "OUT"

    if 0 <= r < 0.333:
        return "A"
    elif 0.333 <= r < 0.519:
        return "B"
    elif 0.519 <= r < 0.815:
        return "C"
    elif 0.815 <= r <= 1.0:
        return "D"
    else:
        return "OUT"



def get_bbox_circle_distance_range(box, cx, cy):
    """
    计算检测框到圆心的最小/最大距离。

    min_dist <= radius 表示检测框与圆内部有交集；
    max_dist >= inner_radius 表示检测框触达外圈区域。
    """
    x1, y1, x2, y2 = [float(v) for v in box]

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    # 圆心到矩形的最小距离：圆心在矩形内部时为 0
    closest_x = min(max(float(cx), x1), x2)
    closest_y = min(max(float(cy), y1), y2)
    min_dist = float(np.sqrt((closest_x - cx) ** 2 + (closest_y - cy) ** 2))

    # 圆心到矩形四个角的最大距离
    corners = np.array([
        [x1, y1], [x1, y2], [x2, y1], [x2, y2]
    ], dtype=np.float64)
    dists = np.sqrt((corners[:, 0] - cx) ** 2 + (corners[:, 1] - cy) ** 2)
    max_dist = float(np.max(dists))

    return min_dist, max_dist


def get_bbox_circle_area_ratios(box, cx, cy, radius, d_inner_ratio=0.815, d_outer_ratio=1.0):
    """
    计算检测框内有多少面积真正落在样品圆内，以及有多少面积落在 D 区环带。

    返回：
    - inside_ratio: 框内位于红色拟合圆内的采样面积占比
    - d_band_ratio: 框内位于 D 区环带的采样面积占比

    说明：
    - 这里用像素采样近似面积，比“只要框碰到圆就算”更稳；
    - 对崩边尤其重要：真实崩边可以跨出圆，但主体不能主要在圆外；
    - D 区外侧严格限制到 1.0*r，不允许把圆外工装盘区域算入 D 区。
    """
    x1, y1, x2, y2 = [float(v) for v in box]

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    box_area = bw * bh

    # 大框用稀疏采样，小框用逐像素采样，兼顾速度和稳定性。
    if box_area > 40000:
        step = 4
    elif box_area > 10000:
        step = 2
    else:
        step = 1

    xs = np.arange(np.floor(x1) + 0.5, np.ceil(x2), step, dtype=np.float32)
    ys = np.arange(np.floor(y1) + 0.5, np.ceil(y2), step, dtype=np.float32)

    if len(xs) == 0 or len(ys) == 0 or radius <= 0:
        return 0.0, 0.0

    xx, yy = np.meshgrid(xs, ys)
    dist = np.sqrt((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2)

    inside = dist <= float(radius)
    d_band = (dist >= d_inner_ratio * float(radius)) & (dist <= d_outer_ratio * float(radius))

    total = float(dist.size)
    inside_ratio = float(np.count_nonzero(inside) / total)
    d_band_ratio = float(np.count_nonzero(d_band) / total)

    return inside_ratio, d_band_ratio


def get_defect_region_ratio(defect_type, box, circle_info):
    """
    计算瑕疵所在区域比例。

    普通瑕疵：仍然按检测框中心点距离判断。

    崩边 chipping：使用“框面积与圆/ D 区环带重叠比例”判断：
      1) 如果框内落在红色拟合圆内的面积太少，说明主体在工装盘/圆外，返回 OUT；
      2) 如果框内有足够面积位于 D 区环带，则归为 D 区；
      3) 如果框虽在圆内但没有触达 D 区，则按中心区域返回，后续规则会忽略。
    """
    if circle_info is None:
        return -1

    cx, cy = circle_info['center']
    radius = float(circle_info['radius'])
    if radius <= 0:
        return -1

    x1, y1, x2, y2 = [float(v) for v in box]
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    center_dist = float(np.sqrt((center_x - cx) ** 2 + (center_y - cy) ** 2))
    center_ratio = center_dist / radius

    if defect_type != 'chipping':
        return center_ratio

    inside_ratio, d_band_ratio = get_bbox_circle_area_ratios(
        (x1, y1, x2, y2),
        cx,
        cy,
        radius,
        d_inner_ratio=0.815,
        d_outer_ratio=1.0,
    )

    # 崩边中心点也必须严格在红色拟合圆内。
    # center_ratio > 1.0 说明框主体已经偏到圆外/工装盘侧，直接按 OUT 丢弃。
    if center_ratio > 1.0:
        return max(1.000001, center_ratio)

    # 崩边框主体必须有足够面积在红色拟合圆内部。
    # 低于该比例时，大概率是工装盘/圆外反光或圆外破损，不上报。
    if inside_ratio < 0.45:
        return max(1.000001, center_ratio)

    # 有足够面积落在 D 区环带，才作为 D 区崩边。
    # D 区外侧边界严格等于拟合圆半径 r，不做 1.02/1.005 之类外扩容忍。
    if d_band_ratio >= 0.20:
        return 0.999999

    # 虽然进入圆内，但没有触达 D 区，按中心点所在区域返回；
    # check_defect_in_region 会把 A/B/C 的 chipping 按误检忽略。
    return center_ratio

def _append_dict_rows_to_csv(csv_path, fieldnames, rows):
    """追加写入 CSV；如果文件不存在或为空，则自动写入表头。"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    need_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_defect_csv_reports(detail_csv_path, summary_csv_path, defect_details,
                            tray_id, pos, part_number, surface_type, image_name):
    """
    保存瑕疵明细 CSV 和 A/B/C/D/OUT 区域统计 CSV。

    detail_csv_path: 每个瑕疵一行
    summary_csv_path: 每个样本/表面/区域一行统计
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tray_id = tray_id if tray_id is not None else ""

    detail_fields = [
        "time", "tray_id", "pos", "part_number", "surface_type", "image_name",
        "defect_no", "defect_uid", "region", "category", "category_id", "confidence",
        "x1", "y1", "x2", "y2", "width_px", "height_px", "max_size_px",
        "region_ratio", "is_qualified", "is_ignored", "reason",
        "weber_contrast", "cnr", "dsnr", "contrast_level"
    ]

    detail_rows = []
    region_stats = {
        r: {"total": 0, "splash": 0, "scratch": 0, "chipping": 0,
            "qualified": 0, "unqualified": 0, "ignored": 0}
        for r in ["A", "B", "C", "D", "OUT"]
    }

    for idx, defect in enumerate(defect_details, start=1):
        bbox = defect.get("bbox", {})
        width = float(bbox.get("width", 0.0))
        height = float(bbox.get("height", 0.0))
        category = defect.get("category", "")
        region_ratio = defect.get("region_ratio", -1)
        region = defect.get("region", get_region_label(region_ratio))
        if region not in region_stats:
            region = "OUT"

        is_qualified = bool(defect.get("is_qualified", True))
        is_ignored = bool(defect.get("is_ignored", False))
        contrast = defect.get("contrast_metrics", {}) or {}

        region_stats[region]["total"] += 1
        if category in ("splash", "scratch", "chipping"):
            region_stats[region][category] += 1
        if is_ignored:
            region_stats[region]["ignored"] += 1
        elif is_qualified:
            region_stats[region]["qualified"] += 1
        else:
            region_stats[region]["unqualified"] += 1

        defect_uid = f"pos{pos}_surface{part_number}_{idx:03d}"
        detail_rows.append({
            "time": now_str,
            "tray_id": tray_id,
            "pos": pos,
            "part_number": part_number,
            "surface_type": surface_type,
            "image_name": image_name,
            "defect_no": idx,
            "defect_uid": defect_uid,
            "region": region,
            "category": category,
            "category_id": defect.get("category_id", ""),
            "confidence": round(float(defect.get("confidence", 0.0)), 6),
            "x1": round(float(bbox.get("x1", 0.0)), 2),
            "y1": round(float(bbox.get("y1", 0.0)), 2),
            "x2": round(float(bbox.get("x2", 0.0)), 2),
            "y2": round(float(bbox.get("y2", 0.0)), 2),
            "width_px": round(width, 2),
            "height_px": round(height, 2),
            "max_size_px": round(max(width, height), 2),
            "region_ratio": round(float(region_ratio), 6) if region_ratio is not None else "",
            "is_qualified": int(is_qualified),
            "is_ignored": int(is_ignored),
            "reason": defect.get("reason", ""),
            "weber_contrast": round(float(contrast.get("weber_contrast", 0.0)), 6) if contrast else "",
            "cnr": round(float(contrast.get("cnr", 0.0)), 6) if contrast else "",
            "dsnr": round(float(contrast.get("dsnr", 0.0)), 6) if contrast else "",
            "contrast_level": contrast.get("contrast_level", "") if contrast else "",
        })

    if detail_rows:
        _append_dict_rows_to_csv(detail_csv_path, detail_fields, detail_rows)

    summary_fields = [
        "time", "tray_id", "pos", "part_number", "surface_type", "image_name", "region",
        "total", "splash", "scratch", "chipping", "qualified", "unqualified", "ignored"
    ]
    summary_rows = []
    for region in ["A", "B", "C", "D", "OUT"]:
        s = region_stats[region]
        summary_rows.append({
            "time": now_str,
            "tray_id": tray_id,
            "pos": pos,
            "part_number": part_number,
            "surface_type": surface_type,
            "image_name": image_name,
            "region": region,
            "total": s["total"],
            "splash": s["splash"],
            "scratch": s["scratch"],
            "chipping": s["chipping"],
            "qualified": s["qualified"],
            "unqualified": s["unqualified"],
            "ignored": s["ignored"],
        })
    _append_dict_rows_to_csv(summary_csv_path, summary_fields, summary_rows)


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
    search_margin=170,
    outer_radius_window=130,
    min_inner_dark=120,
    min_rise_contrast=16,
    min_fall_drop=8,
    inner_search_min_ratio=0.32,
    inner_search_max_ratio=0.56,
    edge_width_min=8,
    edge_width_max=105,
    prefer_edge_width=45,
):
    """
    提取楔形片外径边界点。此版本不再直接围绕理论半径找最强边，
    而是先锁定“样品黑色内部 -> 样品亮边”的内侧边界，
    再只在该亮边后方有限厚度范围内寻找外侧下降边。

    这样可以避免把更外侧的工装盘/融合黑边误认为楔形片外径。

    返回值：
    - outer_points: 楔形片外径候选点
    - inner_points: 与 outer_points 一一对应的亮边内侧候选点
    """

    h, w = gray.shape[:2]

    if init_center is None:
        cx0, cy0 = w / 2.0, h / 2.0
    else:
        cx0, cy0 = init_center

    img = cv2.GaussianBlur(gray, (0, 0), 2.0).astype(np.float32)
    min_hw = min(h, w)

    # 搜索范围要覆盖样品亮边内侧和外侧，但不能只盯着理论半径。
    # 因为融合/裁切后，理论 981.8 px 可能会偏外，容易靠近工装盘。
    r_min = int(max(0, min(inner_search_min_ratio * min_hw, physical_radius_px - search_margin)))
    r_max = int(min(min_hw * 0.60, max(inner_search_max_ratio * min_hw, physical_radius_px + search_margin)))

    angles = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    outer_points = []
    inner_points = []

    for theta in angles:
        rs = np.arange(r_min, r_max, dtype=np.float32)

        xs = cx0 + rs * np.cos(theta)
        ys = cy0 + rs * np.sin(theta)

        valid = (xs >= 1) & (xs < w - 2) & (ys >= 1) & (ys < h - 2)
        if valid.sum() < 100:
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

        if len(profile) < 100:
            continue

        profile_smooth = cv2.GaussianBlur(
            profile.reshape(-1, 1),
            (1, 31),
            0
        ).reshape(-1)

        grad = np.diff(profile_smooth)

        margin = 25
        if len(grad) <= 2 * margin:
            continue

        # -------------------------------------------------
        # Step 1：找样品亮边内侧：黑色样品内部 -> 亮边
        # 关键：选“第一个可信的内侧亮边”，而不是最外侧工装盘边。
        # -------------------------------------------------
        search_idxs = np.arange(margin, len(grad) - margin, dtype=np.int32)

        # 只考虑理论半径附近及其内侧的上升边，避免直接抓到外侧工装结构。
        inner_radius_upper = physical_radius_px + 20
        search_idxs = search_idxs[rs_valid[search_idxs] <= inner_radius_upper]
        if len(search_idxs) == 0:
            continue

        # 取正梯度较强的若干候选，再按半径从小到大检查。
        pos_grad = grad[search_idxs]
        strong = pos_grad > 0
        if not np.any(strong):
            continue

        search_idxs = search_idxs[strong]
        pos_grad = pos_grad[strong]

        # 取前 30 个较强上升边候选，减少噪声误选。
        top_k = min(30, len(search_idxs))
        top_order = np.argsort(pos_grad)[-top_k:]
        candidate_inner_idxs = search_idxs[top_order]
        candidate_inner_idxs = candidate_inner_idxs[np.argsort(rs_valid[candidate_inner_idxs])]

        inner_idx = None
        for idx in candidate_inner_idxs:
            inside_start = max(0, idx - 35)
            inside_end = idx
            after_start = idx + 1
            after_end = min(len(profile_smooth), idx + 36)

            if inside_end <= inside_start or after_end <= after_start:
                continue

            inside_mean = float(profile_smooth[inside_start:inside_end].mean())
            after_mean = float(profile_smooth[after_start:after_end].mean())
            rise_contrast = after_mean - inside_mean

            # 内侧必须是样品暗区；后面必须明显进入亮边。
            if inside_mean > min_inner_dark:
                continue
            if rise_contrast < min_rise_contrast:
                continue

            inner_idx = int(idx)
            break

        if inner_idx is None:
            continue

        r_inner = rs_valid[inner_idx]

        # -------------------------------------------------
        # Step 2：只在样品亮边的有限厚度范围内找外侧下降边。
        # 这个硬约束是防止跳到工装盘的关键。
        # -------------------------------------------------
        fall_start = inner_idx + edge_width_min
        fall_end = min(len(grad) - 5, inner_idx + edge_width_max)

        if fall_end <= fall_start:
            continue

        candidate_idxs = np.arange(fall_start, fall_end, dtype=np.int32)
        candidate_rs = rs_valid[candidate_idxs]

        # 外侧边不能离理论外径太远，但这个约束放宽；真正限制靠 edge_width_max。
        radius_err = np.abs(candidate_rs - physical_radius_px)
        radius_valid = radius_err <= outer_radius_window
        if not np.any(radius_valid):
            continue

        candidate_idxs = candidate_idxs[radius_valid]
        candidate_rs = candidate_rs[radius_valid]
        radius_err = radius_err[radius_valid]

        neg_grad = -grad[candidate_idxs]
        valid_drop = neg_grad > 0
        if not np.any(valid_drop):
            continue

        candidate_idxs = candidate_idxs[valid_drop]
        candidate_rs = candidate_rs[valid_drop]
        radius_err = radius_err[valid_drop]
        neg_grad = neg_grad[valid_drop]

        edge_width = candidate_rs - r_inner

        # 综合评分：下降明显 + 边缘厚度合理 + 不偏离理论外径太多。
        # 不能单纯选择最强负梯度，否则外侧工装盘容易获胜。
        width_penalty = np.abs(edge_width - prefer_edge_width)
        score = neg_grad - 0.12 * width_penalty - 0.03 * radius_err

        # 从得分最高的候选开始检查真实下降幅度。
        order = np.argsort(score)[::-1]
        outer_idx = None

        for oi in order[:10]:
            idx = int(candidate_idxs[oi])
            before_outer = float(profile_smooth[max(0, idx - 20):idx].mean())
            after_outer = float(profile_smooth[idx + 1:min(len(profile_smooth), idx + 31)].mean())
            fall_drop = before_outer - after_outer

            if fall_drop < min_fall_drop:
                continue

            # 如果下降之后几乎是全黑背景，很可能是融合黑边/工装盘外边界，不作为样品外径。
            # 阈值很低，只用于排除纯黑背景。
            if after_outer < 3:
                continue

            outer_idx = idx
            break

        if outer_idx is None:
            continue

        r_outer = rs_valid[outer_idx]

        # 最终硬约束：样品边缘亮带不允许异常厚。
        if not (edge_width_min <= (r_outer - r_inner) <= edge_width_max):
            continue

        x_inner = cx0 + r_inner * np.cos(theta)
        y_inner = cy0 + r_inner * np.sin(theta)
        x_outer = cx0 + r_outer * np.cos(theta)
        y_outer = cy0 + r_outer * np.sin(theta)

        inner_points.append([x_inner, y_inner])
        outer_points.append([x_outer, y_outer])

    outer_points = np.asarray(outer_points, dtype=np.float32)
    inner_points = np.asarray(inner_points, dtype=np.float32)

    return outer_points, inner_points


@timer_decorator()
def detect_and_mask_circle(image):
    """
    使用“内边缘锚定”的楔形片外径拟圆方法检测圆形并输出掩码。
    完全兼容原接口：return masked_img, circle_info

    注意：
    - 先拟合样品亮边内侧圆；
    - 再用一一对应的外侧边界点拟合外径圆；
    - 通过“外径半径 - 内边缘半径”的厚度约束，避免拟到工装盘。
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
            search_margin=170,
            outer_radius_window=130,
            min_inner_dark=120,
            min_rise_contrast=16,
            min_fall_drop=8,
            edge_width_min=8,
            edge_width_max=105,
            prefer_edge_width=45,
        )

        if len(outer_points) < 30 or len(inner_points) < 30:
            return None, None

        # 先拟合亮边内侧圆，用它作为样品位置锚点。
        cx_inner, cy_inner, r_inner_fit, inner_inlier_points = ransac_circle(
            inner_points,
            iterations=3000,
            threshold=10,
            seed=42,
            expected_radius=None,
            expected_center=init_center,
            center_tol=120,
        )

        # 利用内侧圆过滤一一对应的外侧点。
        inner_dist = np.sqrt(
            (inner_points[:, 0] - cx_inner) ** 2 +
            (inner_points[:, 1] - cy_inner) ** 2
        )
        inner_residual = np.abs(inner_dist - r_inner_fit)

        outer_dist_from_inner_center = np.sqrt(
            (outer_points[:, 0] - cx_inner) ** 2 +
            (outer_points[:, 1] - cy_inner) ** 2
        )
        edge_width = outer_dist_from_inner_center - r_inner_fit

        valid_pair = (
            (inner_residual < 15) &
            (edge_width >= 8) &
            (edge_width <= 105)
        )

        outer_points_filtered = outer_points[valid_pair]

        if len(outer_points_filtered) < 30:
            return None, None

        expected_outer_radius = r_inner_fit + float(np.median(edge_width[valid_pair]))

        cx, cy, r_outer_fit, inlier_points = ransac_circle(
            outer_points_filtered,
            iterations=4000,
            threshold=10,
            seed=42,
            expected_radius=expected_outer_radius,
            radius_tol=35,
            expected_center=(cx_inner, cy_inner),
            center_tol=60,
        )

        # 最终厚度检查：如果外径比内边缘大太多，大概率跳到了工装盘。
        final_edge_width = r_outer_fit - r_inner_fit
        if final_edge_width < 5 or final_edge_width > 110:
            print(
                f"外径拟圆厚度异常: inner_r={r_inner_fit:.1f}, "
                f"outer_r={r_outer_fit:.1f}, width={final_edge_width:.1f}"
            )
            return None, None

    except Exception as e:
        print(f"外径拟圆失败: {e}")
        return None, None

    # mask 半径严格等于拟合外径，不再外扩。
    # 这样圆外工装盘区域会在检测前被置黑，减少工装盘崩边误报。
    R_extended = r_outer_fit

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(R_extended)), 255, -1)

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

    # 崩边只允许出现在 D 区；A/B/C/OUT 中的 chipping 按误检忽略，
    # 不参与不合格判定，也不在结果图上画框。
    region_label_for_rule = get_region_label(region_ratio)

    # OUT 区域：样品外区域，所有瑕疵一律忽略，
    # 不参与不合格判定，也不在结果图、CSV、最终上报中输出。
    if region_label_for_rule == "OUT":
        return True, "ignore"

    if defect_type == 'chipping' and region_label_for_rule != "D":
        return True, "ignore"

    if 0 <= region_ratio < 0.333:
        # A区：直径 φ0.9 范围内
        # 图纸要求：
        # 1) 点子 < 0.01 mm，也就是 < 10 μm
        # 2) 划痕宽度 < 0.01 mm，也就是 < 10 μm
        # 3) 划痕长度 < 0.3 mm，也就是 < 300 μm
        # 4) 划痕数量 < 2 条

        spot_threshold = 10 / pixel_factor  # 10 μm -> px
        scratch_width_threshold = 10 / pixel_factor  # 10 μm -> px
        scratch_length_threshold = 300 / pixel_factor  # 300 μm -> px

        if defect_type == 'splash':
            if max_size >= spot_threshold:
                return False, (
                    f"区域A中发现超标麻点"
                    f"（尺寸{width:.1f}x{height:.1f}px，"
                    f"阈值{spot_threshold:.1f}px，约10μm）"
                )

        elif defect_type == 'scratch':
            scratch_length = max(width, height)
            scratch_width = min(width, height)

            # 先判断划痕宽度
            if scratch_width >= scratch_width_threshold:
                return False, (
                    f"区域A中发现超宽划痕"
                    f"（宽度{scratch_width:.1f}px，"
                    f"阈值{scratch_width_threshold:.1f}px，约10μm）"
                )

            # 再判断划痕长度
            if scratch_length >= scratch_length_threshold:
                return False, (
                    f"区域A中发现超长划痕"
                    f"（长度{scratch_length:.1f}px，"
                    f"阈值{scratch_length_threshold:.1f}px，约300μm）"
                )

            # A区划痕数量统计，严格限定在 region_ratio < 0.333
            scratch_count = sum(
                1 for d in all_defects
                if d['type'] == 'scratch'
                and 0 <= d.get('region_ratio', -1) < 0.333
            )

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
            max_threshold = 200 / pixel_factor
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
        self.offline_stop_requested = False
        self.model_configs = {
            "old": {
                "type": "sahi_path",
                "model_path": str(PROJECT_ROOT / "deploy" / "weights" / "best.pt"),
                "conf": 0.25,
            },
            "new": {
                "type": "sahi_yolo_preload",
                "model_path": str(PROJECT_ROOT / "deploy" / "weights" / "best.torchscript"),
                "conf": 0.25,
            }
        }
        self.active_model_key = "old"
        self.sahi_model = None
        self.model_loaded = False
        self.load_defect_model(self.active_model_key)
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.run_result_dir = os.path.join(
            str(RUNTIME_ROOT / "results"),
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

    def read_image_unicode(self, path):
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {path}")
        return img

    def natural_key(self, path):
        return [int(part) if part.isdigit() else part.lower()
                for part in re.split(r'(\d+)', str(path))]

    def classify_camera_file(self, path):
        name = path.stem
        if "焦后" in name or "DA3827093" in name:
            return "DA3827093"
        if "焦面" in name or "DA3562103" in name:
            return "DA3562103"
        if "焦前" in name or "DA3562117" in name:
            return "DA3562117"
        return None

    def parse_online_saved_image_name(self, path):
        name = path.stem
        match = re.search(r'pos\s*(\d+)[_-]+surface\s*([12])[_-]*(.*)$', name, re.IGNORECASE)
        if not match:
            return None
        camera_no = self.classify_camera_file(path)
        if camera_no is None:
            return None
        return int(match.group(1)), int(match.group(2)), camera_no

    def find_offline_surfaces(self, folder):
        folder = Path(folder)
        image_exts = {".jpg", ".jpeg", ".bmp", ".png", ".tif", ".tiff"}
        required = {"DA3827093", "DA3562103", "DA3562117"}

        by_pos_surface = {}
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in image_exts:
                continue
            parsed = self.parse_online_saved_image_name(path)
            if parsed is None:
                continue
            pos, surface, camera_no = parsed
            by_pos_surface.setdefault((pos, surface), {})[camera_no] = path

        surfaces = []
        if by_pos_surface:
            missing_groups = []
            for (pos, surface), files in sorted(by_pos_surface.items()):
                if required.issubset(files):
                    surfaces.append({
                        "pos": pos,
                        "surface": surface,
                        "files": files,
                    })
                else:
                    missing = sorted(required - set(files))
                    missing_groups.append((pos, surface, missing))
            if missing_groups:
                preview = ", ".join(
                    f"pos{pos}_surface{surface}缺{missing}"
                    for pos, surface, missing in missing_groups[:5]
                )
                print(f"[offline] 有 {len(missing_groups)} 个表面不完整，已跳过。示例: {preview}")
            return surfaces

        grouped_by_dir = {}
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in image_exts:
                continue
            camera_no = self.classify_camera_file(path)
            if camera_no is None:
                continue
            grouped_by_dir.setdefault(path.parent, {})[camera_no] = path

        for index, (parent, files) in enumerate(
            sorted(grouped_by_dir.items(), key=lambda item: self.natural_key(item[0])),
            start=1
        ):
            if required.issubset(files):
                surfaces.append({
                    "pos": (index + 1) // 2,
                    "surface": 1 if index % 2 == 1 else 2,
                    "files": files,
                })
        return surfaces

    def start_offline_detect_folder(self, folder, tray_id=None):
        try:
            self.offline_stop_requested = False
            surfaces = self.find_offline_surfaces(folder)
            if not surfaces:
                self.image_ui_queue.put({
                    'type': 'info',
                    'data': 'finish'
                })
                print(f"[offline] 未找到三相机原图组: {folder}")
                return

            complete_surfaces = []
            surfaces_by_pos = {}
            for surface in surfaces:
                surfaces_by_pos.setdefault(surface["pos"], {})[surface["surface"]] = surface
            for pos in sorted(surfaces_by_pos):
                pair = surfaces_by_pos[pos]
                if 1 in pair and 2 in pair:
                    complete_surfaces.extend([pair[1], pair[2]])
                else:
                    print(f"[offline] pos{pos} 缺少 surface1 或 surface2，已跳过。")

            surfaces = complete_surfaces
            if not surfaces:
                self.image_ui_queue.put({'type': 'info', 'data': 'finish'})
                print(f"[offline] 未找到完整样本: {folder}")
                return

            self.Tray_id = tray_id
            self.pending_sample_data = {}
            self.scan_part_counter = 0
            self.final_sample_position = 0

            raw_dir = Path(self.run_result_dir) / "离线原图"
            raw_dir.mkdir(parents=True, exist_ok=True)

            for surface_index, surface in enumerate(surfaces, start=1):
                if self.offline_stop_requested:
                    print("[offline] 已收到停止请求，停止继续提交离线原图。")
                    self.image_ui_queue.put({'type': 'info', 'data': 'finish'})
                    return
                data = {}
                files = surface["files"]
                for camera_no, image_path in files.items():
                    frame = self.read_image_unicode(image_path)
                    data[camera_no] = {
                        'frame': frame,
                        'pic_path': str(raw_dir / f"pos{surface['pos']}_surface{surface['surface']}_{camera_no}{image_path.suffix}")
                    }
                self.debug_image_queue.put(data)

            self.debug_image_queue.put({'type': 'offline_finish'})
            print(f"[offline] 已提交 {len(surfaces)} 个表面，预计 {len(surfaces) // 2} 个样品。")
        except Exception as e:
            print(f"[offline] 离线检测启动失败: {e}")
            import traceback
            traceback.print_exc()
            self.image_ui_queue.put({'type': 'info', 'data': 'finish'})

    def request_offline_stop(self):
        self.offline_stop_requested = True
        self.pending_sample_data = {}
        self.scan_part_counter = 0
        self.final_sample_position = 0
        self.image_ui_queue.put({'type': 'info', 'data': 'finish'})
        print("[offline] 已收到离线检测停止请求。")

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
        model_path_for_loader = model_path
        try:
            model_path_for_loader = os.path.relpath(model_path, PROJECT_ROOT)
        except Exception:
            model_path_for_loader = model_path
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
                    region_ratio = get_defect_region_ratio(
                        defect_type,
                        (minx, miny, maxx, maxy),
                        circle_info
                    )
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
                    region_ratio = get_defect_region_ratio(
                        defect_type,
                        (minx, miny, maxx, maxy),
                        circle_info
                    )

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
                region_label = get_region_label(region_ratio)

                # OUT 区域是样品外区域，直接丢弃：
                # 不画框、不写 CSV、不参与不合格、不进入最终上报。
                if circle_info is not None and region_label == "OUT":
                    continue

                is_ignored = (reason == "ignore")

                # reason == "ignore" 的瑕疵也写入 CSV，方便追溯；
                # 但不作为不合格，也不在结果图上画框。
                if (not is_ignored) and (not is_qualified):
                    unqualified_reasons.append(reason)

                defect_info = {
                    "id": cnt, "category": defect_type, "category_id": int(det[1]),
                    "confidence": float(confidence),
                    "bbox": {
                        "x1": float(minx), "y1": float(miny), "x2": float(maxx),
                        "y2": float(maxy), "width": float(width), "height": float(height)
                    },
                    "region": region_label,
                    "region_ratio": region_ratio,
                    "is_qualified": is_qualified,
                    "is_ignored": is_ignored,
                    "reason": reason
                }
                if contrast_metrics is not None:
                    defect_info['contrast_metrics'] = contrast_metrics
                defect_details.append(defect_info)

                if not is_ignored:
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
                region_label = get_region_label(region_ratio)
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
                cv2.circle(result_img, (center_x, center_y), radius, (0, 0, 255), 2)
                cv2.circle(result_img, (center_x, center_y), 5, (0, 255, 0), -1)
                regions = [0.333, 0.519, 0.815,1]
                colors = [(255, 100, 0), (0, 200, 0), (0, 100, 255), (0, 0, 255)]
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
                if self.offline_stop_requested:
                    if isinstance(data, dict) and data.get('type') == 'offline_finish':
                        print("[offline] 停止后收到 offline_finish，已忽略。")
                    else:
                        print("[offline] 停止请求生效，丢弃后续离线图像处理结果。")
                    continue
                if isinstance(data, dict) and data.get('type') == 'offline_finish':
                    self.pending_sample_data = {}
                    self.scan_part_counter = 0
                    self.final_sample_position = 0
                    self.image_ui_queue.put({'type': 'info', 'data': 'finish'})
                    print("[offline] 离线检测完成。")
                    continue
                part_number = self.scan_part_counter + 1
                surface_type = "下表面" if part_number == 1 else "上表面"
                print(f"图像处理进程接收到样本的第 {part_number} 部分 ({surface_type}) 图像...")

                Image.fromarray(data['DA3562103']['frame']).save(data['DA3562103']['pic_path'])
                img_front_pil = Image.fromarray(data['DA3562117']['frame'])
                img_front_pil.save(data['DA3562117']['pic_path'])
                img_back_pil = Image.fromarray(data['DA3827093']['frame'])
                img_back_pil.save(data['DA3827093']['pic_path'])
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
                fusion_image_list = [img_back, img_focus, img_front]

                pos_for_filename = self.final_sample_position + 1
                pre_fusion_dir = os.path.join(self.fusion_result_dir, "pre_fusion_images")
                os.makedirs(pre_fusion_dir, exist_ok=True)
                img_back_path = os.path.join(pre_fusion_dir, f"pos{pos_for_filename}_surface{part_number}_焦后.jpg")
                cv2.imwrite(img_back_path, img_back)
                img_focus_path = os.path.join(pre_fusion_dir, f"pos{pos_for_filename}_surface{part_number}_焦面.jpg")
                cv2.imwrite(img_focus_path, img_focus)
                img_front_path = os.path.join(pre_fusion_dir, f"pos{pos_for_filename}_surface{part_number}_焦前.jpg")
                cv2.imwrite(img_front_path, img_front)

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

                    # 保存瑕疵 CSV：一个明细表 + 一个按 A/B/C/D/OUT 的区域统计表
                    defect_detail_csv_path = os.path.join(self.result_dir, "defect_detail.csv")
                    defect_summary_csv_path = os.path.join(self.result_dir, "defect_summary.csv")
                    save_defect_csv_reports(
                        detail_csv_path=defect_detail_csv_path,
                        summary_csv_path=defect_summary_csv_path,
                        defect_details=defect_details,
                        tray_id=self.Tray_id if self.Tray_id is not None else "",
                        pos=pos_for_filename,
                        part_number=part_number,
                        surface_type=surface_type,
                        image_name=fusion_result_filename
                    )
                    print(f"瑕疵明细CSV已更新: {defect_detail_csv_path}")
                    print(f"区域统计CSV已更新: {defect_summary_csv_path}")

                    if result_img is not None:
                        cv2.imwrite(current_result_pic_path, result_img)
                    has_unqualified = any((not d.get('is_qualified', True)) and (not d.get('is_ignored', False)) for d in defect_details)
                    if has_unqualified:
                        res = 0
                        for defect in defect_details:
                            if (not defect.get('is_qualified', True)) and (not defect.get('is_ignored', False)):
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
                    result_img = fused_image.copy() if fused_image is not None else None

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
                    elif data['type'] == 'offline_detect_folder':
                        folder = data.get('data', {}).get('folder')
                        tray_id = data.get('data', {}).get('tray_id', self.Tray_id)
                        threading.Thread(
                            target=self.start_offline_detect_folder,
                            args=(folder, tray_id),
                            daemon=True
                        ).start()
                    elif data['type'] == 'stop_offline_detection':
                        self.request_offline_stop()
            except queue.Empty:
                pass
