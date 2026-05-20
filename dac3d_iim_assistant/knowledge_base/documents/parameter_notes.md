# DAC-3D 参数速查

## 相机编号

- `DA3827093`：焦后相机。
- `DA3562103`：焦面相机。
- `DA3562117`：焦前相机。

离线图片命名中可以使用相机编号，也可以使用 `焦后`、`焦面`、`焦前`。

## 模型路径

当前默认瑕疵检测模型为：

```text
deploy/weights/best.pt
```

备用模型路径包括：

```text
deploy/weights/best.torchscript
deploy/model/best.onnx
weights/best.pt
```

主流程默认使用 `old` 模型配置，即 `deploy/weights/best.pt`。

常见问法：默认模型路径是什么、离线模型在哪里、检测模型是哪一个、本地权重文件在哪。当前答案统一为：默认模型是 `deploy/weights/best.pt`，备用模型包括 `deploy/weights/best.torchscript` 和 `deploy/model/best.onnx`。

## 推理参数

- 置信度阈值：`0.25`
- SAHI 切片尺寸：`512 x 512`
- GPU 优先：`cuda:0`
- CPU 回退：支持

## 区域划分

- A 区：`0 <= region_ratio < 0.333`
- B 区：`0.333 <= region_ratio < 0.519`
- C 区：`0.519 <= region_ratio < 0.815`
- D 区：`0.815 <= region_ratio <= 1.0`
- OUT：圆外或无效区域

## 缺陷类型

- `splash`：麻点或飞溅类瑕疵。
- `scratch`：划痕。
- `chipping`：崩边。

## 数据库连接

默认 MySQL 参数：

- host：`localhost`
- port：`3306`
- user：`root`
- password：`123456`
- database：`xxp`

数据库不可用时，系统启用离线数据库模式。

## 结果目录

检测结果默认保存到：

```text
runtime/ftkpic/results/<时间戳>/
```

关键结果文件：

- `检测结果/defect_detail.csv`
- `检测结果/defect_summary.csv`
- `检测结果/*_detect.jpg`
- `融合结果/fused_image_*.jpg`

## 配准矩阵

配准矩阵文件：

```text
Algorithm/Regis_Fusion/homography_config.json
```

其中：

- `H1_to_ref`：焦后到焦面。
- `H3_to_ref`：焦前到焦面。

相机位置或分辨率变化后，应重新标定矩阵。
