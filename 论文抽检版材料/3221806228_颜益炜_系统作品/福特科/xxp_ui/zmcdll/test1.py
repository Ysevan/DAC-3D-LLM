import os
from datetime import datetime

def create_folders(base_path):
    """
    在指定位置创建日期文件夹，并在该文件夹下创建当前时间的子文件夹，
    然后在时间子文件夹下创建焦面、焦前、焦后三个文件夹
    :param base_path: 基础路径，例如 'C:/MyFolder'
    """
    try:
        # 获取当前日期和时间
        now = datetime.now()
        date_folder = now.strftime("%Y-%m-%d")  # 日期文件夹名称，例如 "2023-10-05"
        time_folder = now.strftime("%H-%M-%S")  # 时间文件夹名称，例如 "14-30-00"

        # 创建日期文件夹
        date_folder_path = os.path.join(base_path, date_folder)
        os.makedirs(date_folder_path, exist_ok=True)  # 如果文件夹已存在则忽略

        # 创建时间子文件夹
        time_folder_path = os.path.join(date_folder_path, time_folder)
        os.makedirs(time_folder_path, exist_ok=True)  # 如果文件夹已存在则忽略

        # 在时间子文件夹下创建焦面、焦前、焦后三个文件夹
        folders_to_create = ["焦面", "焦前", "焦后"]
        for folder in folders_to_create:
            folder_path = os.path.join(time_folder_path, folder)
            os.makedirs(folder_path, exist_ok=True)
            print(f"文件夹创建成功: {folder_path}")

        print(f"所有文件夹创建成功: {time_folder_path}")
        return time_folder_path

    except Exception as e:
        print(f"文件夹创建失败: {e}")
        return None

# 示例用法
if __name__ == "__main__":
    base_path = "E:/ftkpic/shexing"  # 指定基础路径
    created_folder = create_folders(base_path)
    if created_folder:
        print(f"最终时间子文件夹路径: {created_folder}")


