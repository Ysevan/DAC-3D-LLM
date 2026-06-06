# import os
# import time
# import argparse
# import cv2
# import numpy as np
# from pathlib import Path

# def load_model(model_path):
#     """
#     根据模型格式加载不同的模型
#     """
#     model_format = Path(model_path).suffix
    
#     if model_format == '.pt':
#         # PyTorch模型
#         from ultralytics import YOLO
#         model = YOLO(model_path)
#         return model, 'pytorch'
    
#     elif model_format == '.onnx':
#         # ONNX模型
#         import onnxruntime as ort
#         providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
#         model = ort.InferenceSession(model_path, providers=providers)
#         return model, 'onnx'
    
#     elif model_format == '.torchscript':
#         # TorchScript模型
#         import torch
#         model = torch.jit.load(model_path)
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         model.to(device)
#         return model, 'torchscript'
    
#     else:
#         raise ValueError(f"不支持的模型格式: {model_format}")

# # def process_image(model, model_type, image_path, conf_threshold=0.25):
# #     """
# #     处理单张图片
# #     """
# #     # 读取图片
# #     img = cv2.imread(image_path)
# #     if img is None:
# #         print(f"无法读取图片: {image_path}")
# #         return None
    
# #     # 根据模型类型进行推理
# #     if model_type == 'pytorch':
# #         # 使用YOLO模型直接推理
# #         results = model(img, conf=conf_threshold)
# #         return results[0].plot()
    
# #     elif model_type == 'onnx':
# #         # ONNX模型推理
# #         img_resized = cv2.resize(img, (640, 640))
# #         img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
# #         # 预处理
# #         input_data = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
# #         input_data = np.expand_dims(input_data, axis=0)
        
# #         # 获取输入输出名称
# #         input_name = model.get_inputs()[0].name
# #         output_names = [output.name for output in model.get_outputs()]
        
# #         # 推理
# #         outputs = model.run(output_names, {input_name: input_data})
        
# #         # 后处理 (简化版，实际应用中需要更复杂的处理)
# #         # 这里假设输出是检测框，需要根据实际模型输出调整
# #         detections = outputs[0]
        
# #         # 绘制结果 (简化版)
# #         result_img = img.copy()
# #         for detection in detections:
# #             if len(detection) >= 6 and detection[4] > conf_threshold:
# #                 x1, y1, x2, y2 = map(int, detection[:4])
# #                 conf = detection[4]
# #                 cls_id = int(detection[5])
                
# #                 # 绘制边界框
# #                 cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #                 cv2.putText(result_img, f"{cls_id}: {conf:.2f}", (x1, y1 - 10),
# #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
# #         return result_img
    
# #     elif model_type == 'torchscript':
# #         # TorchScript模型推理
# #         import torch
        
# #         img_resized = cv2.resize(img, (640, 640))
# #         img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
# #         # 预处理
# #         input_data = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
# #         input_data = input_data.unsqueeze(0)
# #         device = next(model.parameters()).device
# #         input_data = input_data.to(device)
        
# #         # 推理
# #         with torch.no_grad():
# #             outputs = model(input_data)
        
# #         # 后处理 (简化版)
# #         result_img = img.copy()
# #         # 这里需要根据TorchScript模型的输出格式调整
# #         # 由于输出格式可能与原始YOLO不同，这里仅作示例
        
# #         return result_img
# def process_image(model, model_type, image_path, conf_threshold=0.25):
#     """
#     处理单张图片
#     """
#     # 读取图片
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"无法读取图片: {image_path}")
#         return None
    
#     # 根据模型类型进行推理
#     if model_type == 'pytorch':
#         # 使用YOLO模型直接推理，不进行resize
#         results = model(img, conf=conf_threshold)
#         return results[0].plot()
    
#     elif model_type == 'onnx':
#         # ONNX模型推理 - 不进行resize，保持原始尺寸
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # 预处理 - 保持原始尺寸
#         input_data = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
#         input_data = np.expand_dims(input_data, axis=0)
        
#         # 获取输入输出名称
#         input_name = model.get_inputs()[0].name
#         output_names = [output.name for output in model.get_outputs()]
        
#         # 推理
#         outputs = model.run(output_names, {input_name: input_data})
        
#         # 后处理
#         detections = outputs[0]
        
#         # 绘制结果
#         result_img = img.copy()
#         for detection in detections:
#             if len(detection) >= 6 and detection[4] > conf_threshold:
#                 x1, y1, x2, y2 = map(int, detection[:4])
#                 conf = detection[4]
#                 cls_id = int(detection[5])
                
#                 # 绘制边界框
#                 cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(result_img, f"{cls_id}: {conf:.2f}", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
#         return result_img
    
#     elif model_type == 'torchscript':
#         # TorchScript模型推理 - 不进行resize，保持原始尺寸
#         import torch
        
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # 预处理 - 保持原始尺寸
#         input_data = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
#         input_data = input_data.unsqueeze(0)
#         device = next(model.parameters()).device
#         input_data = input_data.to(device)
        
#         # 推理
#         with torch.no_grad():
#             outputs = model(input_data)
        
#         # 后处理
#         result_img = img.copy()
#         # 这里需要根据TorchScript模型的输出格式调整
        
#         return result_img




# def main():
#     # 使用交互式方式设置参数，避免命令行参数解析问题
#     class Args:
#         def __init__(self):
#             self.model = r"C:\Users\Administrator\Desktop\yolov11\ultralytics\deployment\exported_models\best.pt"
#             self.input = r"C:\Users\Administrator\Desktop\yolov11\ultralytics\TestImg\pic154.bmp"
#             self.output = r"C:\Users\Administrator\Desktop\yolov11\ultralytics\result"
#             self.conf = 0.25
            

#     args = Args()
    
#     # 创建输出目录
#     if not os.path.exists(args.output):
#         os.makedirs(args.output)
    
#     # 加载模型
#     print(f"加载模型: {args.model}")
#     model, model_type = load_model(args.model)
#     print(f"模型类型: {model_type}")
    
#     # 处理输入
#     if os.path.isdir(args.input):
#         # 处理文件夹
#         image_files = [f for f in os.listdir(args.input) 
#                       if f.lower().endswith(('png', 'jpg', 'bmp', 'jpeg'))]
        
#         total_time = 0
#         num_images = len(image_files)
        
#         if num_images == 0:
#             print(f"警告: 在文件夹 {args.input} 中没有找到图片文件")
#             return
            
#         print(f"找到 {num_images} 张图片，开始处理...")
        
#         for image_file in image_files:
#             image_path = os.path.join(args.input, image_file)
            
#             # 记录开始时间
#             start_time = time.time()
            
#             # 处理图片
#             result_img = process_image(model, model_type, image_path, args.conf)
            
#             if result_img is not None:
#                 # 保存结果
#                 output_path = os.path.join(args.output, f"result_{image_file}")
#                 cv2.imwrite(output_path, result_img)
                
#                 # 计算处理时间
#                 elapsed_time = time.time() - start_time
#                 total_time += elapsed_time
                
#                 print(f"处理图片 {image_file} 耗时: {elapsed_time:.2f} 秒")
        
#         # 输出总耗时和平均处理时间
#         print(f"\n整个批量检测过程总耗时: {total_time:.2f} 秒")
#         print(f"每张图片平均处理时间: {total_time / num_images:.2f} 秒")
    
#     else:
#         # 处理单张图片
#         start_time = time.time()
        
#         result_img = process_image(model, model_type, args.input, args.conf)

#         print(f"\n模型的类型为:{model_type}\n")
        
#         if result_img is not None:
#             # 保存结果
#             output_path = os.path.join(args.output, f"result_{os.path.basename(args.input)}")
#             cv2.imwrite(output_path, result_img)
            
#             # 计算处理时间
#             elapsed_time = time.time() - start_time
#             print(f"处理图片耗时: {elapsed_time:.2f} 秒")

# if __name__ == "__main__":
#     main()


# ----------------------：以下为version2.0，没有使用sahi的推理----------------------
    

# import os
# import time
# import argparse
# import cv2
# import numpy as np
# from pathlib import Path
# import torch
# from sahi import AutoDetectionModel
# from sahi.predict import get_sliced_prediction
# from ultralytics.utils.plotting import Annotator, colors

# def load_model(model_path):
#     """
#     根据模型格式加载不同的模型
#     """
#     model_format = Path(model_path).suffix
    
#     if model_format == '.pt':
#         # 使用SAHI的AutoDetectionModel加载模型
#         model = AutoDetectionModel.from_pretrained(
#             model_type='yolov8',
#             model_path=model_path,
#             confidence_threshold=0.25,
#             device="cuda:0" if torch.cuda.is_available() else "cpu"
#         )
#         return model, 'sahi'
    
#     elif model_format == '.onnx':
#         # ONNX模型
#         import onnxruntime as ort
#         providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
#         model = ort.InferenceSession(model_path, providers=providers)
#         return model, 'onnx'
    
#     elif model_format == '.torchscript':
#         # TorchScript模型
#         model = torch.jit.load(model_path)
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         model.to(device)
#         return model, 'torchscript'
    
#     else:
#         raise ValueError(f"不支持的模型格式: {model_format}")

# def process_image_with_sahi(model, image_path, conf_threshold=0.25, slice_height=2048, slice_width=2048):
#     """
#     使用SAHI进行切片检测处理
#     """
#     # 读取图片
#     frame = cv2.imread(image_path)
#     if frame is None:
#         print(f"无法读取图片: {image_path}")
#         return None, None, 0
    
#     # 创建标注器
#     annotator = Annotator(frame)
    
#     # 使用SAHI进行切片预测
#     result = get_sliced_prediction(
#         image_path,
#         model,
#         slice_height=slice_height,
#         slice_width=slice_width,
#         overlap_height_ratio=0.2,
#         overlap_width_ratio=0.2
#     )
    
#     # 提取检测数据
#     detection_data = [
#         (det.category.name, det.category.id, (det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy))
#         for det in result.object_prediction_list
#     ]
    
#     # 统计麻点数量
#     cnt = 0
#     for det in detection_data:
#         cnt += 1
#         # 解析边界框坐标
#         minx, miny, maxx, maxy = det[2]

#         # 计算边界框的宽度和高度
#         width = maxx - minx
#         height = maxy - miny

#         # 格式化宽度和高度，保留两位小数
#         formatted_width = "{:.2f}".format(width)
#         formatted_height = "{:.2f}".format(height)

#         # 打印检测到的对象的信息，包括边界框的宽度和高度
#         print(
#             print(f"Category: {det[0]}, 数量：{cnt}, ID: {det[1]}, "
#               f"Bounding Box: ({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f}), "
#               f"Width: {formatted_width}, Height: {formatted_height}")
#         )

#         # 使用annotator在图像上绘制边界框和标签
#         annotator.box_label(det[2], label=str(det[0]), color=colors(int(det[1]), True))
    
#     # 获取标注后的图像
#     result_img = annotator.result()
    
#     return result_img, result, cnt

# def process_image(model, model_type, image_path, conf_threshold=0.25, slice_height=2048, slice_width=2048):
#     """
#     处理单张图片
#     """
#     if model_type == 'sahi':
#         # 使用SAHI进行切片检测
#         return process_image_with_sahi(model, image_path, conf_threshold, slice_height, slice_width)
    
#     elif model_type == 'onnx':
#         # ONNX模型推理
#         # ... 原有的ONNX处理代码 ...
#         return None, None, 0
    
#     elif model_type == 'torchscript':
#         # TorchScript模型推理
#         # ... 原有的TorchScript处理代码 ...
#         return None, None, 0
    
#     else:
#         print(f"不支持的模型类型: {model_type}")
#         return None, None, 0

# def main():
#     # 使用交互式方式设置参数，避免命令行参数解析问题
#     class Args:
#         def __init__(self):
#             self.model = r"C:\Users\Administrator\Desktop\yolov11\ultralytics\deployment\exported_models\best.pt"
#             self.input = r"C:\Users\Administrator\Desktop\yolov11\ultralytics\TestImg\pic154.bmp"
#             self.output = r"C:\Users\Administrator\Desktop\yolov11\ultralytics\result"
#             self.conf = 0.25
#             self.slice_height = 1024
#             self.slice_width = 1024
    
#     args = Args()
    
#     # 创建输出目录
#     if not os.path.exists(args.output):
#         os.makedirs(args.output)
    
#     # 加载模型
#     print(f"加载模型: {args.model}")
#     model, model_type = load_model(args.model)
#     print(f"模型类型: {model_type}")
    
#     # 处理输入
#     if os.path.isdir(args.input):
#         # 处理文件夹
#         image_files = [f for f in os.listdir(args.input) 
#                       if f.lower().endswith(('png', 'jpg', 'bmp', 'jpeg'))]
        
#         total_time = 0
#         total_objects = 0
#         num_images = len(image_files)
        
#         if num_images == 0:
#             print(f"警告: 在文件夹 {args.input} 中没有找到图片文件")
#             return
            
#         print(f"找到 {num_images} 张图片，开始处理...")
        
#         for image_file in image_files:
#             image_path = os.path.join(args.input, image_file)
            
#             # 记录开始时间
#             start_time = time.time()
            
#             # 处理图片
#             result_img, sahi_result, object_count = process_image(
#                 model, model_type, image_path, args.conf, args.slice_height, args.slice_width
#             )
            
#             total_objects += object_count
            
#             if result_img is not None:
#                 # 保存结果图像
#                 output_path = os.path.join(args.output, f"result_{image_file}")
#                 cv2.imwrite(output_path, result_img)
                
#                 # 如果有SAHI结果，导出可视化结果
#                 if sahi_result is not None:
#                     sahi_output_dir = os.path.join(args.output, "sahi_visuals")
#                     if not os.path.exists(sahi_output_dir):
#                         os.makedirs(sahi_output_dir)
#                     sahi_result.export_visuals(export_dir=sahi_output_dir)
                
#                 # 计算处理时间
#                 elapsed_time = time.time() - start_time
#                 total_time += elapsed_time
                
#                 print(f"处理图片 {image_file} 耗时: {elapsed_time:.2f} 秒，检测到 {object_count} 个麻点")
        
#         # 输出总耗时和平均处理时间
#         print(f"\n整个批量检测过程总耗时: {total_time:.2f} 秒")
#         print(f"每张图片平均处理时间: {total_time / num_images:.2f} 秒")
#         print(f"总共检测到 {total_objects} 个麻点")
    
#     else:
#         # 处理单张图片
#         start_time = time.time()
        
#         result_img, sahi_result, object_count = process_image(
#             model, model_type, args.input, args.conf, args.slice_height, args.slice_width
#         )
        
#         if result_img is not None:
#             # 保存结果图像
#             output_path = os.path.join(args.output, f"result_{os.path.basename(args.input)}")
#             cv2.imwrite(output_path, result_img)
            
#             # 如果有SAHI结果，导出可视化结果
#             if sahi_result is not None:
#                 sahi_output_dir = os.path.join(args.output, "sahi_visuals")
#                 if not os.path.exists(sahi_output_dir):
#                     os.makedirs(sahi_output_dir)
#                 sahi_result.export_visuals(export_dir=sahi_output_dir)
            
#             # 计算处理时间
#             elapsed_time = time.time() - start_time
#             print(f"处理图片耗时: {elapsed_time:.2f} 秒，检测到 {object_count} 个麻点")

# if __name__ == "__main__":
#     main()




# ----version3.0,使用annotator在图像上绘制边界框和标签,使用NMS后处理

import os
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics.utils.plotting import Annotator, colors
def load_model(model_path):
    """
    根据模型格式加载不同的模型
    """
    model_format = Path(model_path).suffix
    if model_format == '.pt':
        # 使用SAHI的AutoDetectionModel加载模型
        model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=model_path,
            confidence_threshold=0.25,
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        return model, 'sahi'
    elif model_format == '.onnx':
        # ONNX模型
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        model = ort.InferenceSession(model_path, providers=providers)
        return model, 'onnx'
    elif model_format == '.torchscript':
        # TorchScript模型
        model = torch.jit.load(model_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        return model, 'torchscript'
    else:
        raise ValueError(f"不支持的模型格式: {model_format}")

def process_image_with_sahi(model, image_path, conf_threshold=0.25, slice_height=1024, slice_width=1024):
    """
    使用SAHI进行切片检测处理，优化版本
    """
    # 读取图片
    frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if frame is None:
        print(f"无法读取图片: {image_path}")
        return None, None, 0
    # 创建标注器
    annotator = Annotator(frame, line_width=2)  # 增加线宽提高可视性
    # 使用SAHI进行切片预测，优化参数
    result = get_sliced_prediction(
        image_path,
        model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=0.1,  # 减少重叠区域提高速度
        overlap_width_ratio=0.1,   # 减少重叠区域提高速度
        perform_standard_pred=True,  # 先进行标准预测
        postprocess_type="NMS",      # 使用NMS后处理
        postprocess_match_threshold=0.5,  # NMS阈值
        postprocess_match_metric="IOU"    # 使用IOU作为匹配度量
    )
    # 提取检测数据
    detection_data = [
        (det.category.name, det.category.id, (det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy))
        for det in result.object_prediction_list
    ]
    # 统计麻点数量
    cnt = 0
    for det in detection_data:
        cnt += 1
        # 解析边界框坐标
        minx, miny, maxx, maxy = det[2]
        # 计算边界框的宽度和高度
        width = maxx - minx
        height = maxy - miny
        # 格式化宽度和高度，保留两位小数
        formatted_width = "{:.2f}".format(width)
        formatted_height = "{:.2f}".format(height)
        # 打印检测到的对象的信息，包括边界框的宽度和高度
        print(
            print(f"Category: {det[0]}, 数量：{cnt}, ID: {det[1]}, "
              f"Bounding Box: ({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f}), "
              f"Width: {formatted_width}, Height: {formatted_height}")
        )
        # 使用annotator在图像上绘制边界框和标签
        annotator.box_label(det[2], label=str(det[0]), color=colors(int(det[1]), True))
    # 获取标注后的图像
    result_img = annotator.result()
    return result_img, result, cnt

def main():
    # 使用交互式方式设置参数，避免命令行参数解析问题
    class Args:
        def __init__(self):
            self.model = r"C:\Users\Administrator\Desktop\yolov11\ultralytics\deployment\exported_models\best.pt"
            self.input = r"C:\Users\Administrator\Desktop\yolov11\ultralytics\TestImg\pic154.bmp"
            self.output = r"C:\Users\Administrator\Desktop\yolov11\ultralytics\result"
            self.conf = 0.25
            self.slice_height = 1024  # 减小切片尺寸提高速度
            self.slice_width = 1024   # 减小切片尺寸提高速度
    # 其余代码保持不变

def process_image(model, model_type, image_path, conf_threshold=0.25, slice_height=2048, slice_width=2048):
    """
    处理单张图片
    """
    if model_type == 'sahi':
        # 使用SAHI进行切片检测
        return process_image_with_sahi(model, image_path, conf_threshold, slice_height, slice_width)
    elif model_type == 'onnx':
        # ONNX模型推理
        # ... 原有的ONNX处理代码 ...
        return None, None, 0
    elif model_type == 'torchscript':
        # TorchScript模型推理
        # ... 原有的TorchScript处理代码 ...
        return None, None, 0
    else:
        print(f"不支持的模型类型: {model_type}")
        return None, None, 0

def main():
    # 使用交互式方式设置参数，避免命令行参数解析问题
    class Args:
        def __init__(self):
            self.model = r"C:\Users\Administrator\Desktop\yolov11\ultralytics\deployment\exported_models\best.pt"
            self.input = r"C:\Users\Administrator\Desktop\yolov11\ultralytics\TestImg\pic154.bmp"
            self.output = r"C:\Users\Administrator\Desktop\yolov11\ultralytics\result"
            self.conf = 0.25
            self.slice_height = 1024
            self.slice_width = 1024
    args = Args()
    # 创建输出目录
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    # 加载模型
    print(f"加载模型: {args.model}")
    model, model_type = load_model(args.model)
    print(f"模型类型: {model_type}")
    # 处理输入
    if os.path.isdir(args.input):
        # 处理文件夹
        image_files = [f for f in os.listdir(args.input) 
                      if f.lower().endswith(('png', 'jpg', 'bmp', 'jpeg'))]
        total_time = 0
        total_objects = 0
        num_images = len(image_files)
        if num_images == 0:
            print(f"警告: 在文件夹 {args.input} 中没有找到图片文件")
            return           
        print(f"找到 {num_images} 张图片，开始处理...")      
        for image_file in image_files:
            image_path = os.path.join(args.input, image_file)
            # 记录开始时间
            start_time = time.time()
            # 处理图片
            result_img, sahi_result, object_count = process_image(
                model, model_type, image_path, args.conf, args.slice_height, args.slice_width
            )
            total_objects += object_count
            if result_img is not None:
                # 保存结果图像
                output_path = os.path.join(args.output, f"result_{image_file}")
                cv2.imwrite(output_path, result_img)
                # 如果有SAHI结果，导出可视化结果
                if sahi_result is not None:
                    sahi_output_dir = os.path.join(args.output, "sahi_visuals")
                    if not os.path.exists(sahi_output_dir):
                        os.makedirs(sahi_output_dir)
                    sahi_result.export_visuals(export_dir=sahi_output_dir)                
                # 计算处理时间
                elapsed_time = time.time() - start_time
                total_time += elapsed_time               
                print(f"处理图片 {image_file} 耗时: {elapsed_time:.2f} 秒，检测到 {object_count} 个麻点")        
        # 输出总耗时和平均处理时间
        print(f"\n整个批量检测过程总耗时: {total_time:.2f} 秒")
        print(f"每张图片平均处理时间: {total_time / num_images:.2f} 秒")
        print(f"总共检测到 {total_objects} 个麻点")   
    else:
        # 处理单张图片
        start_time = time.time()
        result_img, sahi_result, object_count = process_image(
            model, model_type, args.input, args.conf, args.slice_height, args.slice_width
        )        
        if result_img is not None:
            # 保存结果图像
            output_path = os.path.join(args.output, f"result_{os.path.basename(args.input)}")
            cv2.imwrite(output_path, result_img)            
            # 如果有SAHI结果，导出可视化结果
            if sahi_result is not None:
                sahi_output_dir = os.path.join(args.output, "sahi_visuals")
                if not os.path.exists(sahi_output_dir):
                    os.makedirs(sahi_output_dir)
                sahi_result.export_visuals(export_dir=sahi_output_dir)           
            # 计算处理时间
            elapsed_time = time.time() - start_time
            print(f"处理图片耗时: {elapsed_time:.2f} 秒，检测到 {object_count} 个麻点")
if __name__ == "__main__":
    main()