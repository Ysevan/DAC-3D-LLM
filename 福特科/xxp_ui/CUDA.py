# 方法1：PyTorch 检查
import torch
print("PyTorch 支持 CUDA：", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA 版本：", torch.version.cuda)
    print("显卡名称：", torch.cuda.get_device_name(0))
else:
    print("未检测到 CUDA 支持")

