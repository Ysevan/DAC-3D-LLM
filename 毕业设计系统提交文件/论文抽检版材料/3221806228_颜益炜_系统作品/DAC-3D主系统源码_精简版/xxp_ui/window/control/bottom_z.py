import pandas as pd

# 将 Excel 另存为 CSV，然后用 pandas 读取
df = pd.read_csv(r"C:\Users\Administrator\Desktop\工装盘焦面11111111.csv",header=None)
b_column_data = df.iloc[:, 1].tolist()
c_data= df.iloc[:, 2].tolist()
print(b_column_data)  # 输出 B 列数据（Python 列表）
print(c_data)  # 输出 B 列数据（Python 列表）

