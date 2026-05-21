import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def tenengrad(image):
    """优化后的Tenengrad清晰度计算函数"""
    if len(image.shape) == 3:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_img = image.copy()

    # 使用分离的Sobel计算提高精度
    sobel_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    sobel_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)

    # 优化梯度计算（避免边缘误差）
    G = np.sqrt(sobel_x[2:-2, 2:-2] ** 2 + sobel_y[2:-2, :-2] ** 2)  # 裁剪5%边缘
    return G.mean()  # 使用均值代替总和，便于跨图比较


def process_folder(folder_path, start_distance=0, step=1):
    """处理文件夹中的图像序列"""
    files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.bmp'))],
                   key=lambda x: int(os.path.splitext(x)[0]))

    results = []
    for i, filename in enumerate(tqdm(files, desc="Processing Images")):
        img_path = os.path.join(folder_path, filename)
        try:
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("Invalid image")

            clarity = tenengrad(img)
            axial_distance = start_distance + i * step
            results.append({
                "filename": filename,
                "distance": axial_distance,
                "tenengrad": round(clarity, 4)
            })
        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")

    return pd.DataFrame(results)


def visualize_results(df):
    """可视化清晰度-距离曲线"""
    plt.figure(figsize=(12, 6))
    plt.plot(df['distance'], df['tenengrad'], 'b-o', linewidth=1, markersize=5)
    plt.title('Tenengrad Sharpness vs Axial Distance')
    plt.xlabel('Axial Distance (mm)')
    plt.ylabel('Tenengrad Sharpness')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sharpness_curve.png')
    plt.close()


if __name__ == "__main__":
    # 配置参数
    IMAGE_FOLDER = "path/to/your/images"  # 替换为实际路径
    START_DISTANCE = 0  # 起始轴向距离（单位：mm）
    DISTANCE_STEP = 0.1  # 相邻图片的轴向距离间隔（单位：mm）

    # 处理图像
    df = process_folder(IMAGE_FOLDER, START_DISTANCE, DISTANCE_STEP)

    # 保存结果
    df.to_csv('sharpness_results.csv', index=False)
    print("\nSaved results to sharpness_results.csv")

    # 生成曲线
    visualize_results(df)
    print("Generated sharpness_curve.png")