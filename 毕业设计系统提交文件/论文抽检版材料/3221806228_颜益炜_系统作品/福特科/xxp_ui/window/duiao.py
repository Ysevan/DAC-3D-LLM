import cv2
import numpy as np
from scipy import optimize
import os

# 基础环境配置
os.environ["OPENCV_VIDEOIO_BACKEND"] = "MSMF"


class LightSpotAnalyzer:
    """
    激光光斑分析器
    公式: y = 48.33x^2 - 1289.47x + 8871.10 (基于像素面积拟合)
    """

    def __init__(self, px_to_mm=0.0877, target_xy=(1197, 666)):
        """
        初始化参数
        """
        self.px_to_mm = px_to_mm
        self.target_xy = target_xy

        # --- 屏蔽区域设置 ---
        self.mask_width = 500
        self.mask_height = 500

        # --- 拟合系数 ---
        # 对应公式: y = 48.33x^2 - 1289.47x + 8871.10
        self.area_coeffs = [48.33, -1289.47, 8871.10]

        self.intensity_coeffs = self.area_coeffs.copy()

        # --- 图像处理参数 ---
        self.min_brightness = 40
        self.min_area_pixels = 10
        self.roi_padding = 60

    def get_fitting_y(self, x, coeffs):
        """ 计算多项式的值 """
        return np.polyval(coeffs, x)

    def analyze_image(self, img, use_roi=False, roi_x=None, roi_y=None, visualize=False):
        """
        核心分析接口
        Returns: area(px), intensity, x_sol_area, x_sol_intensity
        """
        if img is None: return 0, 0, 0, 0

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # [A] 屏蔽左上角
        h_img, w_img = gray.shape
        mask_h = min(self.mask_height, h_img)
        mask_w = min(self.mask_width, w_img)
        gray[0:mask_h, 0:mask_w] = 0

        # [B] 粗略定位
        blur_search = cv2.GaussianBlur(gray, (21, 21), 0)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blur_search)

        if max_val < self.min_brightness:
            return 0, 0, 0, 0

        cx_rough, cy_rough = max_loc

        # [C] 自动 ROI
        x1 = max(0, cx_rough - self.roi_padding)
        y1 = max(0, cy_rough - self.roi_padding)
        x2 = min(w_img, cx_rough + self.roi_padding)
        y2 = min(h_img, cy_rough + self.roi_padding)
        roi_img = gray[y1:y2, x1:x2]

        # [D] 精细分割
        roi_max = roi_img.max()
        thresh_val = max(roi_max * 0.5, self.min_brightness)
        _, binary = cv2.threshold(roi_img, thresh_val, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return 0, 0, 0, 0

        main_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(main_contour)
        if M["m00"] == 0: return 0, 0, 0, 0

        # === 关键修改：直接使用像素面积 ===
        pixel_area = M["m00"]

        if pixel_area < self.min_area_pixels: return 0, 0, 0, 0

        # [E] 计算物理属性 (不再转换为 mm²)
        # real_area_mm = pixel_area * (self.px_to_mm ** 2)

        mask = np.zeros_like(roi_img)
        cv2.drawContours(mask, [main_contour], -1, 255, -1)
        intensity = cv2.mean(roi_img, mask=mask)[0]

        cx_local = M["m10"] / M["m00"]
        cx_global = cx_local + x1

        # [F] 解算离焦量 X
        target_x = self.target_xy[0]

        # 使用求根公式 ax^2 + bx + (c - y) = 0
        # 这里的 coeffs 本身就是基于像素面积拟合的，所以直接用 pixel_area 计算是正确的
        a, b, c = self.area_coeffs
        c_prime = c - pixel_area
        delta = b ** 2 - 4 * a * c_prime

        x_sol_area = 0.0

        if delta >= 0:
            # 两个解
            x1_sol = (-b - np.sqrt(delta)) / (2 * a)
            x2_sol = (-b + np.sqrt(delta)) / (2 * a)

            # 根据光斑与 Target 的相对位置判断取哪个解
            if cx_global < target_x:
                x_sol_area = x1_sol  # 较小的解
            else:
                x_sol_area = x2_sol  # 较大的解
        else:
            # 无解（面积小于最小值），取顶点
            x_sol_area = -b / (2 * a)

        x_sol_int = 0.0

        # [G] 可视化
        if visualize:
            self._visualize_debug(img, main_contour, (x1, y1), cx_global, x_sol_area, (mask_w, mask_h))

        # === 返回像素面积 ===
        return pixel_area, intensity, x_sol_area, x_sol_int

    def _visualize_debug(self, img, contour, offset, cx, x_sol, mask_size):
        if len(img.shape) == 2:
            res = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            res = img.copy()

        mw, mh = mask_size
        cv2.rectangle(res, (0, 0), (mw, mh), (60, 60, 60), -1)
        x1, y1 = offset
        cv2.rectangle(res, (x1, y1), (x1 + 120, y1 + 120), (0, 255, 255), 1)
        cnt_global = contour + np.array(offset)
        cv2.drawContours(res, [cnt_global], -1, (0, 0, 255), 2)
        cv2.circle(res, (int(cx), int(cnt_global[:, :, 1].mean())), 5, (0, 255, 0), -1)
        cv2.putText(res, f"Pos: {x_sol:.2f}mm", (10, mh + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Analyzer Debug", res)
        cv2.waitKey(1)


if __name__ == "__main__":
    analyzer = LightSpotAnalyzer()
    # 请修改为您的测试图片路径
    test_img_path = r"E:\ftkpic\layerscan\2025-12-26\14-40-00\12.0.bmp"

    if os.path.exists(test_img_path):
        # 注意：这里读取路径如果是中文路径可能需要用之前提到的 cv_imread 辅助函数
        # 为了保持代码独立性，这里直接演示用 cv2.imread，如遇读取失败请替换回之前的 cv_imread
        img = cv2.imread(test_img_path)

        area, intensity, x_val, x_int = analyzer.analyze_image(img, visualize=True)

        print(f"面积: {area:.2f} pixels")  # 单位已改为 pixels
        print(f"计算位置: {x_val:.4f} mm")
    else:
        print("图片路径不存在")
