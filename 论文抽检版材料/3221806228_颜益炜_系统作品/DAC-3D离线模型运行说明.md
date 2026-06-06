# DAC-3D离线模型运行说明

抽检包中的 DAC-3D 主系统只要求复现离线模型运行，不要求连接真实相机、运动控制卡或在线扫描硬件。

## 已保留内容

- 离线检测主流程源码：`image_processor_22.py`
- 默认离线检测权重：`deploy/weights/best.pt`
- 离线推理相关脚本：`deploy/inference.py`、`deploy/infer/infer_withflow.py`
- 一组完整示例图：`DAC-3D离线检测示例数据/pre_fusion_images`

## 运行方式

1. 进入 `DAC-3D主系统源码_精简版/xxp_ui`。
2. 安装 PyQt5、opencv-python、numpy、Pillow、torch、ultralytics、sahi 等依赖。
3. 执行 `python main.py`。
4. 在界面中选择示例图目录，启动离线检测。

如评阅环境缺少 GPU，程序可按代码中的设备选择逻辑回退到 CPU，但推理速度会变慢。
