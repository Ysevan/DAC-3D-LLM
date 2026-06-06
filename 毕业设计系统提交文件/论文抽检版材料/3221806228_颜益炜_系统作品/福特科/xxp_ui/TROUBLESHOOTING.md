# 模型加载问题快速修复指南

**问题**: 程序显示"警告: 模型未加载或融合失败，使用随机结果进行模拟。"

---

## 🔍 第一步：运行诊断脚本

在项目目录下运行诊断脚本：

```bash
cd D:\zycgit\ZDevelop_Confocal\xxp_ui
python check_model.py
```

诊断脚本会检查：
- ✅ Python 版本和路径
- ✅ 必要的依赖包
- ✅ CUDA、Apple MPS、MLX 环境状态
- ✅ 模型文件存在性
- ✅ 模型加载和推理测试

---

## 🔧 常见问题及解决方案

### 问题 1: ultralytics 未安装或版本过低

**症状**:
```
❌ Ultralytics YOLO 未安装
```

**解决方案**:
```bash
pip install ultralytics>=8.0.0
# 或者升级
pip install --upgrade ultralytics
```

### 问题 2: PyTorch 未安装

**症状**:
```
❌ PyTorch 未安装
```

**解决方案**:
```bash
# CPU 版本
pip install torch torchvision

# NVIDIA GPU 版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Apple Silicon Mac 上使用 PyTorch MPS 时，安装普通 PyTorch 后运行：

```bash
python CUDA.py
```

如果输出 `离线推理设备： mps`，说明检测链路会使用 PyTorch MPS；如果输出 `coreml`，说明会使用现有 ONNX 模型和 ONNX Runtime `CoreMLExecutionProvider`；如果输出 `cpu`，说明当前环境没有暴露可用 Apple 加速后端。可用 `DAC3D_INFERENCE_DEVICE=mps` 显式请求 MPS，`DAC3D_INFERENCE_DEVICE=coreml` 显式请求 ONNX/CoreML，或用 `DAC3D_INFERENCE_DEVICE=cpu` 强制 CPU。`mlx` 目前只作为 Apple 加速请求识别，检测链路没有原生 MLX 模型后端。

macOS 26.x 上如果出现类似下面的输出：

```text
MPS 已编译： True
MPS 可用： False
MPS 不可用原因： torch.backends.mps.is_available() 返回 False: The MPS backend is supported on macOS 14.0+...
```

这表示 PyTorch 的 native MPS 初始化失败，不是业务代码里设备名写错。此时不要强制把检测模型移动到 `mps`。程序会优先尝试 ONNX Runtime 的 `CoreMLExecutionProvider` 路线；如果 CoreML EP 也不可用，才回退 CPU。CoreML 不等同于 PyTorch MPS，也不是 MLX，但在 Apple Silicon 上可以使用 Apple 的 GPU/ANE/CPU 计算单元。

在受限沙盒或某些 CI 环境里，MPS 可能误报不可用；请以普通终端或 PyQt 主程序环境中的 `python CUDA.py` 输出为准。本机在原生权限下 `torch 2.12.0` 可以正常创建 `mps` 张量并完成 SAHI/YOLO 小图推理。

### 问题 3: 模型文件路径错误

**症状**:
```
❌ 模型文件不存在
路径: D:\zycgit\ZDevelop_Confocal\xxp_ui\weights\best.pt
```

**解决方案**:
1. 检查模型文件是否存在
2. 确认路径是否正确
3. 如果文件在其他位置，修改 `image_processor.py` 中的路径：
   ```python
   model_path = r"你的模型文件路径\best.pt"
   ```

### 问题 4: 模型文件损坏

**症状**:
```
✅ 模型文件存在
❌ 加载失败: [错误信息]
```

**解决方案**:
1. 重新下载或复制模型文件
2. 检查文件完整性（大小应该在 10MB 左右）

---

## 📝 查看程序启动日志

程序启动时会显示详细的模型加载信息：

### ✅ 成功的日志示例：
```
正在加载瑕疵检测模型...
  模型路径: D:\zycgit\ZDevelop_Confocal\xxp_ui\weights\best.pt
  使用设备: cuda:0
✅ 瑕疵检测模型加载成功！
  模型类别: ['splash', 'scratch', 'chipping']
```

### ❌ 失败的日志示例：
```
正在加载瑕疵检测模型...
  模型路径: D:\zycgit\ZDevelop_Confocal\xxp_ui\weights\best.pt
❌ 加载瑕疵检测模型失败: [错误信息]
   错误类型: [错误类型]
```

**检查点**:
1. 程序启动时，找到上述日志
2. 如果看到 ❌，查看具体的错误信息
3. 根据错误信息采取相应的解决方案

---

## 🚀 验证修复

修复后，再次运行程序，应该看到：

```
正在加载瑕疵检测模型...
  模型路径: D:\zycgit\ZDevelop_Confocal\xxp_ui\weights\best.pt
  使用设备: cuda:0
✅ 瑕疵检测模型加载成功！
  模型类别: ['splash', 'scratch', 'chipping']

...

检查前置条件:
  模型已加载: True
  融合图像: 成功

[上表面] 开始圆检测和瑕疵检测...
[上表面] 步骤1: 开始圆形检测和拟合...
```

---

## 🆘 如果问题仍未解决

### 1. 收集信息

运行诊断脚本并保存输出：
```bash
python check_model.py > diagnostic_report.txt
```

### 2. 检查环境

确认使用的是正确的 Python 环境：
```bash
python --version
python -c "import sys; print(sys.executable)"
```

### 3. 完整重装依赖

```bash
pip uninstall ultralytics torch torchvision -y
pip install torch torchvision ultralytics
```

### 4. 提供日志

将以下信息提供给技术支持：
- `diagnostic_report.txt` 的内容
- 程序启动时的完整控制台输出
- 错误信息的完整堆栈跟踪

---

## 📦 完整的依赖安装命令

如果要重新安装所有依赖：

```bash
# 基础依赖
pip install numpy opencv-python pillow

# PyTorch (NVIDIA GPU 版本)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Ultralytics YOLO
pip install ultralytics>=8.0.0

# 其他依赖
pip install matplotlib pandas
```

---

## ✅ 检查清单

在报告问题之前，请确认：

- [ ] 运行了 `check_model.py` 诊断脚本
- [ ] 确认所有依赖包已安装
- [ ] 确认模型文件存在且完整
- [ ] 查看了程序启动时的完整日志
- [ ] 尝试了完整重装依赖
- [ ] 确认使用的是正确的 Python 环境

---

**最后更新**: 2025-12-18
