import cv2
import numpy as np
from pathlib import Path
import re
from scipy import optimize
import matplotlib.pyplot as plt
import os  # 新增：用于环境配置

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 新增：配置OpenCV视频后端（修复GUI问题）
os.environ["OPENCV_VIDEOIO_BACKEND"] = "MSMF"  # Windows媒体基础后端


class LightSpotAnalyzer:
    """光斑检测和分析类"""

    def __init__(self, px_to_mm=0.0877):
        """
        初始化分析器
        Parameters:
        -----------
        px_to_mm : float, 像素到毫米的转换比例
        """
        self.px_to_mm = px_to_mm

        # 面积拟合曲线系数：y = 0.1556x⁴ + 0.1354x³ + 9.9747x² + 2.1542x + 4.6881
        self.area_coeffs = {
            'a4':-71.31,  # x⁴系数
            'a3':1076.18,  # x³系数
            'a2': 5871.35,  # x²系数
            'a1': 13616.5,  # x系数
            'a0': 11042.3  # 常数项（修复原代码中缺失的右括号）
        }

        # 光强拟合曲线系数（假设与面积相同，实际应替换为真实系数）
        self.intensity_coeffs = self.area_coeffs.copy()

    def print_fitting_curve_info(self):
        """打印拟合曲线信息"""
        print("===== 光斑面积拟合曲线 =====")
        print(f"面积拟合公式: y = {self.area_coeffs['a4']}x⁴  + {self.area_coeffs['a3']}x³  + "
              f"{self.area_coeffs['a2']}x²  + {self.area_coeffs['a1']}x  + {self.area_coeffs['a0']}")
        print("\n===== 光强拟合曲线 =====")
        print(f"光强拟合公式: y = {self.intensity_coeffs['a4']}x⁴  + {self.intensity_coeffs['a3']}x³  + "
              f"{self.intensity_coeffs['a2']}x²  + {self.intensity_coeffs['a1']}x  + {self.intensity_coeffs['a0']}")

    def area_fitting_function(self, x):
        """面积拟合函数"""
        return (self.area_coeffs['a4'] * x ** 4 +
                self.area_coeffs['a3'] * x ** 3 +
                self.area_coeffs['a2'] * x ** 2 +
                self.area_coeffs['a1'] * x +
                self.area_coeffs['a0'])

    def intensity_fitting_function(self, x):
        """光强拟合函数"""
        return (self.intensity_coeffs['a4'] * x ** 4 +
                self.intensity_coeffs['a3'] * x ** 3 +
                self.intensity_coeffs['a2'] * x ** 2 +
                self.intensity_coeffs['a1'] * x +
                self.intensity_coeffs['a0'])

    def select_roi_with_mouse(self, image):
        """
        使用鼠标交互选择ROI区域
        Parameters:
            image: 输入图像
        Returns:
            roi: 选中的ROI区域 (x1, y1, x2, y2)
        """
        # 新增：确保窗口正确初始化
        cv2.namedWindow(' 选择光斑区域 (按Enter确认，Esc取消)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow(' 选择光斑区域 (按Enter确认，Esc取消)', 800, 600)  # 调整窗口大小以便查看
        cv2.startWindowThread()  # 启动窗口线程

        # 选择ROI（修复原代码中参数错误，使用正确的窗口名称）
        roi = cv2.selectROI(
            '选择光斑区域 (按Enter确认，Esc取消)',  # 窗口名称需与namedWindow一致
            image,
            showCrosshair=True,
            fromCenter=False
        )
        cv2.destroyWindow(' 选择光斑区域 (按Enter确认，Esc取消)')
        cv2.waitKey(1)  # 处理窗口关闭事件

        return roi

    def analyze_single_image(self, img_path, use_roi=True, roi_x=None, roi_y=None, visualize=True):
        """
        分析单张图像
        Parameters:
            img_path: 图像路径
            use_roi: 是否使用ROI区域
            roi_x: 预设ROI的x范围 (x1, x2)
            roi_y: 预设ROI的y范围 (y1, y2)
            visualize: 是否可视化结果
        Returns:
            area: 光斑面积
            intensity: 光斑强度
            x_sol_area: 面积对应的x解
            x_sol_intensity: 强度对应的x解
        """
        # 读取图像
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {img_path}")

            # 转换为灰度图（如果是彩色图）
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

            # 选择ROI区域
        if use_roi:
            if roi_x is None or roi_y is None:
                # 交互式选择ROI
                print("请在图像中框选光斑区域...")
                roi = self.select_roi_with_mouse(gray)
                x1, y1, w, h = roi
                x2, y2 = x1 + w, y1 + h
            else:
                # 使用预设ROI
                x1, x2 = roi_x
                y1, y2 = roi_y

                # 裁剪ROI区域（修复原代码中索引错误）
            roi_img = gray[y1:y2, x1:x2]  # 正确的索引顺序是 [y1:y2, x1:x2]
        else:
            roi_img = gray.copy()
            x1, y1, x2, y2 = 0, 0, gray.shape[1], gray.shape[0]

            # 计算光斑面积（假设二值化阈值为127，可根据实际图像调整）
        _, binary = cv2.threshold(roi_img, 127, 255, cv2.THRESH_BINARY)
        area = cv2.countNonZero(binary) * self.px_to_mm ** 2  # 转换为实际面积

        # 计算光斑强度（区域内平均灰度值）
        intensity = np.mean(roi_img[binary == 255]) if cv2.countNonZero(binary) > 0 else 0

        # 求解拟合曲线方程（面积）
        def area_eq(x):
            return self.area_fitting_function(x) - area

        x_sol_area = optimize.fsolve(area_eq, x0=1.0)[0]  # 初始猜测值x0=1.0

        # 求解拟合曲线方程（强度）
        def intensity_eq(x):
            return self.intensity_fitting_function(x) - intensity

        x_sol_intensity = optimize.fsolve(intensity_eq, x0=1.0)[0]

        # 可视化
        if visualize:
            self.visualize_results(
                original_img=img,
                gray_img=gray,
                roi_img=roi_img,
                binary_img=binary,
                roi_coords=(x1, y1, x2, y2),
                area=area,
                intensity=intensity,
                x_sol_area=x_sol_area,
                x_sol_intensity=x_sol_intensity
            )

        return area, intensity, x_sol_area, x_sol_intensity
    def analyze_image(self, image, use_roi=False, roi_x=None, roi_y=None, visualize=True):
        """
        直接分析已加载的图像对象（cv2.imread 读取的结果）
        Parameters:
            image: numpy.ndarray 图像对象
            use_roi: 是否使用ROI区域
            roi_x, roi_y: ROI坐标范围
            visualize: 是否显示可视化结果
        Returns:
            area, intensity, x_sol_area, x_sol_intensity
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("无效的图像对象，请确保已正确加载（cv2.imread 返回的图像）")

        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # ROI 选择
        if use_roi:
            if roi_x is None or roi_y is None:
                print("请在图像中框选光斑区域...")
                roi = self.select_roi_with_mouse(gray)
                x1, y1, w, h = roi
                x2, y2 = x1 + w, y1 + h
            else:
                x1, x2 = roi_x
                y1, y2 = roi_y
            roi_img = gray[y1:y2, x1:x2]
        else:
            roi_img = gray.copy()
            x1, y1, x2, y2 = 0, 0, gray.shape[1], gray.shape[0]

        # 二值化
        _, binary = cv2.threshold(roi_img, 127, 255, cv2.THRESH_BINARY)
        area = cv2.countNonZero(binary) * self.px_to_mm ** 2
        intensity = np.mean(roi_img[binary == 255]) if cv2.countNonZero(binary) > 0 else 0

        # 面积与强度求解
        def area_eq(x):
            return self.area_fitting_function(x) - area
        x_sol_area = optimize.fsolve(area_eq, x0=1.0)[0]

        def intensity_eq(x):
            return self.intensity_fitting_function(x) - intensity
        x_sol_intensity = optimize.fsolve(intensity_eq, x0=1.0)[0]

        if visualize:
            self.visualize_results(
                original_img=image,
                gray_img=gray,
                roi_img=roi_img,
                binary_img=binary,
                roi_coords=(x1, y1, x2, y2),
                area=area,
                intensity=intensity,
                x_sol_area=x_sol_area,
                x_sol_intensity=x_sol_intensity
            )

        return area, intensity, x_sol_area, x_sol_intensity
    #
    # def visualize_results(self, original_img, gray_img, roi_img, binary_img, roi_coords, area, intensity, x_sol_area,
    #                       x_sol_intensity):
    #     """可视化分析结果"""
    #     x1, y1, x2, y2 = roi_coords
    #
    #     # 创建结果图像（在原图上绘制ROI）
    #     result_img = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR) if len(
    #         original_img.shape) == 2 else original_img.copy()
    #     cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色矩形标记ROI
    #
    #     # 显示基本信息
    #     info_text = [
    #         f"面积: {area:.2f} mm² (x={x_sol_area:.4f})",
    #         f"强度: {intensity:.2f} (x={x_sol_intensity:.4f})"
    #     ]
    #     for i, text in enumerate(info_text):
    #         cv2.putText(
    #             result_img, text, (10, 30 + i * 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
    #         )
    #
    #         # 绘制拟合曲线
    #     self.plot_fitting_curves(area, intensity, x_sol_area, x_sol_intensity)
    #
    #     # 显示图像
    #     cv2.namedWindow(' 分析结果', cv2.WINDOW_NORMAL)
    #     cv2.resizeWindow(' 分析结果', 1000, 800)
    #     cv2.imshow(' 分析结果', result_img)
    #
    #     # 显示ROI和二值化结果
    #     cv2.namedWindow('ROI 区域', cv2.WINDOW_NORMAL)
    #     cv2.imshow('ROI 区域', roi_img)
    #     cv2.namedWindow(' 二值化结果', cv2.WINDOW_NORMAL)
    #     cv2.imshow(' 二值化结果', binary_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plot_fitting_curves(self, area, intensity, x_sol_area, x_sol_intensity):
        """绘制拟合曲线和求解点"""
        plt.figure(figsize=(12, 5))

        # 面积拟合曲线
        plt.subplot(1, 2, 1)
        x = np.linspace(0, max(x_sol_area * 1.2, 0.1), 100)  # 生成x值范围
        y_area = self.area_fitting_function(x)
        plt.plot(x, y_area, 'b-', label='面积拟合曲线')
        plt.scatter(x_sol_area, area, color='red', s=100, zorder=5, label=f'求解点 ({x_sol_area:.4f}, {area:.2f})')
        plt.xlabel('x')
        plt.ylabel(' 面积 (mm²)')
        plt.title(' 光斑面积拟合曲线与求解点')
        plt.legend()
        plt.grid(True)

        # 强度拟合曲线
        plt.subplot(1, 2, 2)
        y_intensity = self.intensity_fitting_function(x)
        plt.plot(x, y_intensity, 'g-', label='强度拟合曲线')
        plt.scatter(x_sol_intensity, intensity, color='red', s=100, zorder=5,
                    label=f'求解点 ({x_sol_intensity:.4f}, {intensity:.2f})')
        plt.xlabel('x')
        plt.ylabel(' 强度')
        plt.title(' 光斑强度拟合曲线与求解点')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


def main():
    """主函数"""
    analyzer = LightSpotAnalyzer(px_to_mm=0.0877)
    analyzer.print_fitting_curve_info()

    # 图像路径（请替换为您的图像路径）
    img_path = r"E:\ftkpic\layerscan\2025-12-25\21-07-11\1.9.bmp"  # 保持您原有的图像路径

    # 分析图像（使用交互式ROI选择）
    area, intensity, x_sol_area, x_sol_intensity = analyzer.analyze_single_image(
        img_path,
        use_roi=False,
        visualize=True  # 启用可视化
    )

    # 打印结果
    print("\n===== 分析结果 =====")
    print(f"光斑面积: {area:.4f} mm²")
    print(f"光斑强度: {intensity:.4f}")
    print(f"面积对应x值: {x_sol_area:.6f}")
    print(f"强度对应x值: {x_sol_intensity:.6f}")


if __name__ == "__main__":
    main()