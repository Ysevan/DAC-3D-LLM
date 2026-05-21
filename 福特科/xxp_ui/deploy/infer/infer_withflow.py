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

# def process_image_with_sahi(model, image_path, conf_threshold=0.25, slice_height=1024, slice_width=1024):
#     """
#     使用SAHI进行切片检测处理，优化版本
#     """
#     # 读取图片
#     frame = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
#     if frame is None:
#         print(f"无法读取图片: {image_path}")
#         return None, None, 0
#     # 创建标注器
#     annotator = Annotator(frame, line_width=2)  # 增加线宽提高可视性
#     # 使用SAHI进行切片预测，优化参数
#     result = get_sliced_prediction(
#         image_path,
#         model,
#         slice_height=slice_height,
#         slice_width=slice_width,
#         overlap_height_ratio=0.1,  # 减少重叠区域提高速度
#         overlap_width_ratio=0.1,   # 减少重叠区域提高速度
#         perform_standard_pred=True,  # 先进行标准预测
#         postprocess_type="NMS",      # 使用NMS后处理
#         postprocess_match_threshold=0.5,  # NMS阈值
#         postprocess_match_metric="IOU"    # 使用IOU作为匹配度量
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

# def main():
#     # 使用交互式方式设置参数，避免命令行参数解析问题
#     class Args:
#         def __init__(self):
#             self.model = r"C:\Users\Administrator\Desktop\yolov11\ultralytics\deployment\exported_models\best.pt"
#             self.input = r"C:\Users\Administrator\Desktop\yolov11\ultralytics\TestImg\pic154.bmp"
#             self.output = r"C:\Users\Administrator\Desktop\yolov11\ultralytics\result"
#             self.conf = 0.25
#             self.slice_height = 1024  # 减小切片尺寸提高速度
#             self.slice_width = 1024   # 减小切片尺寸提高速度
#     # 其余代码保持不变

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


import os
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
import torch
import json
from datetime import datetime
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
        return None, None, 0, []
    
    # 创建标注器 - 修改线宽为2
    annotator = Annotator(frame, line_width=2)
    
    # 使用SAHI进行切片预测，优化参数
    result = get_sliced_prediction(
        image_path,
        model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=0.1,
        overlap_width_ratio=0.1,
        perform_standard_pred=True,
        postprocess_type="NMS",
        postprocess_match_threshold=0.5,
        postprocess_match_metric="IOU"
    )
    
    # 提取检测数据，包括置信度分数
    detection_data = [
        (det.category.name, det.category.id, (det.bbox.minx, det.bbox.miny, det.bbox.maxx, det.bbox.maxy), float(det.score.value))
        for det in result.object_prediction_list
    ]
    
    # 统计麻点数量
    cnt = 0
    defect_details = []
    
    for det in detection_data:
        cnt += 1
        # 解析边界框坐标和置信度
        minx, miny, maxx, maxy = det[2]
        confidence = det[3]
        
        # 计算边界框的宽度和高度
        width = maxx - minx
        height = maxy - miny
        
        # 记录缺陷详细信息
        defect_info = {
            "id": cnt,
            "category": det[0],
            "category_id": int(det[1]),
            "confidence": float(confidence),
            "bbox": {
                "x1": float(minx),
                "y1": float(miny),
                "x2": float(maxx),
                "y2": float(maxy),
                "width": float(width),
                "height": float(height)
            }
        }
        defect_details.append(defect_info)
        
        # 打印检测到的对象的信息
        print(f"Category: {det[0]}, 数量：{cnt}, ID: {det[1]}, Confidence: {confidence:.2f}, "
              f"Bounding Box: ({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f}), "
              f"Width: {width:.2f}, Height: {height:.2f}")
        
        # 修改标签格式为 "splash 0.XX"，使用红色作为标签颜色
        label = f"{det[0]} {confidence:.2f}"
        # 使用红色 (0,0,255) 作为边界框和标签颜色
        annotator.box_label(det[2], label=label, color=(0, 0, 128))
    
    # 获取标注后的图像
    result_img = annotator.result()
    
    return result_img, result, cnt, defect_details
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
        return None, None, 0, []
    elif model_type == 'torchscript':
        # TorchScript模型推理
        # ... 原有的TorchScript处理代码 ...
        return None, None, 0, []
    else:
        print(f"不支持的模型类型: {model_type}")
        return None, None, 0, []

def batch_process_folder(input_dir, output_dir, model_path, conf_threshold=0.25, slice_height=1024, slice_width=1024):
    """
    批量处理文件夹中的所有图像
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建JSON结果目录
    json_output_dir = os.path.join(output_dir, "json_results")
    if not os.path.exists(json_output_dir):
        os.makedirs(json_output_dir)
    
    # 创建SAHI可视化结果目录
    sahi_output_dir = os.path.join(output_dir, "sahi_visuals")
    if not os.path.exists(sahi_output_dir):
        os.makedirs(sahi_output_dir)
    
    # 加载模型
    print(f"加载模型: {model_path}")
    model, model_type = load_model(model_path)
    print(f"模型类型: {model_type}")
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('png', 'jpg', 'bmp', 'jpeg'))]
    
    total_time = 0
    total_objects = 0
    num_images = len(image_files)
    
    if num_images == 0:
        print(f"警告: 在文件夹 {input_dir} 中没有找到图片文件")
        return
    
    print(f"找到 {num_images} 张图片，开始处理...")
    
    # 创建汇总结果
    summary = {
        "total_images": num_images,
        "total_defects": 0,
        "total_time": 0,
        "average_time": 0,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": "",
        "results": []
    }
    
    # 处理每张图像
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        
        # 记录开始时间
        start_time = time.time()
        
        # 处理图片
        result_img, sahi_result, object_count, defect_details = process_image(
            model, model_type, image_path, conf_threshold, slice_height, slice_width
        )
        
        total_objects += object_count
        
        if result_img is not None:
            # 保存结果图像
            output_path = os.path.join(output_dir, f"result_{image_file}")
            cv2.imwrite(output_path, result_img)
            
            # 如果有SAHI结果，导出可视化结果
            if sahi_result is not None:
                sahi_result.export_visuals(export_dir=sahi_output_dir)
            
            # 计算处理时间
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            # 保存JSON结果
            result_info = {
                "image_path": image_path,
                "image_name": image_file,
                "processing_time": elapsed_time,
                "defect_count": object_count,
                "defect_details": defect_details,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            }
            
            json_path = os.path.join(json_output_dir, f"{os.path.splitext(image_file)[0]}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_info, f, ensure_ascii=False, indent=2)
            
            # 更新汇总信息
            summary["results"].append({
                "image_name": image_file,
                "defect_count": object_count,
                "processing_time": elapsed_time
            })
            
            print(f"处理图片 {image_file} 耗时: {elapsed_time:.2f} 秒，检测到 {object_count} 个麻点")
    
    # 更新汇总结果
    summary["total_defects"] = total_objects
    summary["total_time"] = total_time
    summary["average_time"] = total_time / num_images if num_images > 0 else 0
    summary["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 保存汇总结果
    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n整个批量检测过程总耗时: {total_time:.2f} 秒")
    print(f"每张图片平均处理时间: {total_time / num_images:.2f} 秒")
    print(f"总共检测到 {total_objects} 个麻点")
    
    return total_objects, total_time

def main():
    """
    主函数，用于批量处理指定文件夹下的所有图片
    """
    # 设置参数
    # D:\zycgit\ZDevelop_Confocal\xxp_ui\deploy\infer\IMG\TestImg
    input_dir = r"D:\zycgit\ZDevelop_Confocal\xxp_ui\deploy\infer\IMG\TestImg"  # 输入文件夹路径
    output_dir = r"D:\zycgit\ZDevelop_Confocal\xxp_ui\deploy\infer\IMG\ResImg"  # 输出文件夹路径
    model_path = r"D:\zycgit\ZDevelop_Confocal\xxp_ui\deploy\exported_models\best.pt"  # 模型路径
    conf_threshold = 0.3# 置信度阈值
    slice_height = 1024  # 切片高度
    slice_width = 1024  # 切片宽度
    
    # 批量处理文件夹
    batch_process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        model_path=model_path,
        conf_threshold=conf_threshold,
        slice_height=slice_height,
        slice_width=slice_width
    )

if __name__ == "__main__":
    main()
