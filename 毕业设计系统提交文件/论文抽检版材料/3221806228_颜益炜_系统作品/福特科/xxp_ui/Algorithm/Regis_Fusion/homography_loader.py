"""
旋转矩阵配置加载工具模块
用于从配置文件中加载图像配准的变换矩阵
"""
import json
import numpy as np
import os


def load_homography_matrices(config_path=None):
    """
    从配置文件加载旋转矩阵

    参数:
        config_path: 配置文件路径，默认为同目录下的 homography_config.json

    返回:
        tuple: (H1_to_ref, H3_to_ref) 两个numpy数组形式的变换矩阵
    """
    if config_path is None:
        # 默认配置文件路径
        config_path = os.path.join(
            os.path.dirname(__file__),
            "homography_config.json"
        )

    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"警告: 配置文件不存在: {config_path}")
        print("使用默认的旋转矩阵")
        # 返回默认矩阵（单位矩阵）
        return np.eye(3), np.eye(3)

    try:
        # 读取JSON配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 将列表转换为numpy数组
        H1_to_ref = np.array(config['H1_to_ref'])
        H3_to_ref = np.array(config['H3_to_ref'])

        print(f"成功加载旋转矩阵配置")
        print(f"配置时间: {config.get('timestamp', '未知')}")

        return H1_to_ref, H3_to_ref

    except Exception as e:
        print(f"错误: 加载配置文件失败: {e}")
        print("使用默认的旋转矩阵")
        return np.eye(3), np.eye(3)


def get_homographies_list(config_path=None):
    """
    获取包含三个矩阵的列表 [H1_to_ref, np.eye(3), H3_to_ref]

    参数:
        config_path: 配置文件路径

    返回:
        list: 包含三个变换矩阵的列表
    """
    H1_to_ref, H3_to_ref = load_homography_matrices(config_path)
    return [H1_to_ref, np.eye(3), H3_to_ref]


if __name__ == "__main__":
    # 测试代码
    print("测试旋转矩阵加载...")
    H1, H3 = load_homography_matrices()
    print("\nH1_to_ref (焦后到焦面):")
    print(H1)
    print("\nH3_to_ref (焦前到焦面):")
    print(H3)
