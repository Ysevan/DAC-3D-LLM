import torch

from device_compat import coreml_status, mlx_available, mps_status, select_inference_device

device = select_inference_device("auto")
mps = mps_status()
coreml = coreml_status()

print("PyTorch 支持 CUDA：", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA 版本：", torch.version.cuda)
    print("显卡名称：", torch.cuda.get_device_name(0))
else:
    print("未检测到 CUDA 支持")
print("MPS 已编译：", mps.built)
print("MPS 可用：", mps.available)
if mps.reason:
    print("MPS 不可用原因：", mps.reason)
print("MLX 可用：", mlx_available())
print("CoreML EP 可用：", coreml.available)
if coreml.reason:
    print("CoreML EP 不可用原因：", coreml.reason)
print("离线推理设备：", device.name)
print("离线推理加速器：", device.label)
