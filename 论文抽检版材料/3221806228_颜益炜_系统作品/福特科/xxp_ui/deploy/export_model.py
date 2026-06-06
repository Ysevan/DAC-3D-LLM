def export_yolo_model(model_path, export_dir, formats=None):
    """
    导出YOLOv8模型为多种格式
    
    参数:
        model_path: 模型路径
        export_dir: 导出目录
        formats: 导出格式列表，默认为['onnx', 'torchscript', 'engine']
    """
    if formats is None:
        formats = ['onnx', 'torchscript', 'engine']
    
    # 创建导出目录
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
    
    # 检测CUDA是否可用
    import torch
    cuda_available = torch.cuda.is_available()
    device = "0" if cuda_available else "cpu"
    
    # 加载模型
    model = YOLO(model_path)
    
    # 导出为不同格式
    for format_type in formats:
        print(f"正在导出 {format_type} 格式...")
        try:
            if format_type == 'engine':
                if not cuda_available:
                    print(f"跳过 {format_type} 格式导出: 未检测到CUDA设备，TensorRT需要NVIDIA GPU支持")
                    continue
                # TensorRT 导出需要特殊配置
                model.export(format=format_type, imgsz=640, device=device, half=True)
            else:
                model.export(format=format_type, imgsz=640, device=device)
            print(f"成功导出 {format_type} 格式")
        except Exception as e:
            print(f"导出 {format_type} 格式失败: {e}")
            if format_type == 'engine':
                print("TensorRT导出失败，请确保已正确安装CUDA、cuDNN和TensorRT")
    
    # 复制原始PT模型到导出目录
    shutil.copy(model_path, export_dir)
    
    print(f"模型已导出到: {export_dir}")
    return True