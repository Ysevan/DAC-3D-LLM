import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from matplotlib.backend_bases import PickEvent

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class InteractiveGrayPlot:
    def __init__(self, folder_path, roi):
        self.folder_path = Path(folder_path)
        self.roi = roi
        self.df = None
        self.fig, self.ax = plt.subplots(figsize=(14, 7))
        self.annotation = None

    def process_images(self):
        """处理所有图片并提取ROI数据"""
        x1, y1, x2, y2 = self.roi
        image_files = sorted([f for f in self.folder_path.glob('*')
                              if f.suffix.lower() in ('.bmp', '.jpg', '.png')],
                             key=lambda x: x.name)

        if not image_files:
            raise ValueError("未找到支持的图片文件")

        print(f"处理 {len(image_files)} 张图片...")

        data = []
        for img_path in tqdm(image_files, desc="处理进度"):
            try:
                img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                h, w = img.shape
                if w < x2 or h < y2:
                    continue

                roi_region = img[y1:y2, x1:x2]
                data.append({
                    'filepath': str(img_path),
                    'filename': img_path.name,
                    'gray_value': np.mean(roi_region),
                    'roi_data': roi_region
                })
            except Exception as e:
                print(f"\n处理失败 {img_path.name}: {str(e)}")
                continue

        if not data:
            raise ValueError("没有成功处理任何图片")

        self.df = pd.DataFrame(data)
        return self.df

    def plot(self):
        """绘制交互式图表"""
        if self.df is None:
            self.process_images()

        # 绘制主趋势线（启用点选功能）
        self.line, = self.ax.plot(
            self.df.index,
            self.df['gray_value'],
            'o-',
            color='#1f77b4',
            linewidth=2,
            markersize=8,
            picker=True,  # 启用点选
            pickradius=10  # 点选敏感度
        )

        # 添加平均线
        mean_val = self.df['gray_value'].mean()
        self.ax.axhline(mean_val, color='#d62728', linestyle='--',
                        linewidth=1.5, label=f'平均值: {mean_val:.1f}')

        # 图表装饰
        self.ax.set_title('ROI灰度值变化趋势（点击数据点查看详情）', fontsize=16, pad=20)
        self.ax.set_xlabel('图像序列号', fontsize=14)
        self.ax.set_ylabel('灰度值', fontsize=14)
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.ax.legend(fontsize=12)

        # 智能X轴标签
        if len(self.df) > 30:
            step = max(1, len(self.df) // 10)
            xticks = range(0, len(self.df), step)
            self.ax.set_xticks(xticks)
            self.ax.set_xticklabels([f"图{i + 1}" for i in xticks], rotation=45)
        else:
            self.ax.set_xticks(range(len(self.df)))
            self.ax.set_xticklabels([f"图{i + 1}" for i in range(len(self.df))], rotation=45)

        # 连接事件处理器
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_hover)

        plt.tight_layout()
        plt.show()

    def on_pick(self, event):
        """点击数据点时触发的回调函数"""
        if event.artist != self.line:
            return

        ind = event.ind[0]
        row = self.df.iloc[ind]

        # 清除旧标注
        if self.annotation:
            self.annotation.remove()

        # 创建新标注
        self.annotation = self.ax.annotate(
            f"文件: {row['filename']}\n灰度值: {row['gray_value']:.2f}",
            xy=(ind, row['gray_value']),
            xytext=(20, 20),
            textcoords='offset points',
            bbox=dict(boxstyle='round', fc='w', alpha=0.9),
            arrowprops=dict(arrowstyle='->')
        )

        # 显示ROI区域图像
        self.show_roi_image(row)

        self.fig.canvas.draw_idle()

    def on_hover(self, event):
        """鼠标悬停时高亮数据点"""
        if not event.inaxes == self.ax:
            return

        contains, ind = self.line.contains(event)
        if contains:
            self.line.set_marker('*')
            self.line.set_markersize(12)
            self.line.set_markeredgecolor('red')
        else:
            self.line.set_marker('o')
            self.line.set_markersize(8)
            self.line.set_markeredgecolor('#1f77b4')

        self.fig.canvas.draw_idle()

    def show_roi_image(self, row):
        """显示ROI区域图像"""
        if hasattr(self, 'roi_fig'):
            plt.close(self.roi_fig)

        self.roi_fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # 显示原始图像（缩略图）
        img = cv2.imread(row['filepath'])
        if img is not None:
            h, w = img.shape[:2]
            scale = 300 / max(h, w)
            small_img = cv2.resize(img, (int(w * scale), int(h * scale)))
            ax[0].imshow(cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB))
            ax[0].set_title('原始图像缩略图')

            # 在缩略图上标记ROI位置
            x1, y1, x2, y2 = self.roi
            rect = plt.Rectangle(
                (x1 * scale, y1 * scale),
                (x2 - x1) * scale,
                (y2 - y1) * scale,
                linewidth=2,
                edgecolor='r',
                facecolor='none'
            )
            ax[0].add_patch(rect)

        # 显示ROI区域放大图
        ax[1].imshow(row['roi_data'], cmap='gray', vmin=0, vmax=255)
        ax[1].set_title(f'ROI区域 (灰度值: {row["gray_value"]:.2f})')

        plt.tight_layout()
        plt.show()


def main():
    # 配置参数
    image_folder = r"E:\ftkpic\layerscan\2025-04-16\10-56-11"
    roi_coords = (1081, 469, 1094, 480)  # (x1, y1, x2, y2)

    try:
        # 创建交互式图表
        plotter = InteractiveGrayPlot(image_folder, roi_coords)
        plotter.process_images()

        # 保存数据
        output_csv = "灰度值分析结果.csv"
        plotter.df.drop('roi_data', axis=1).to_csv(output_csv, index=False, encoding='utf_8_sig')
        print(f"\n数据已保存到: {output_csv}")

        # 显示交互式图表
        plotter.plot()

    except Exception as e:
        print(f"程序出错: {str(e)}")


if __name__ == '__main__':
    main()