import os
import cv2
import torch
import ultralytics
import sahi
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions

def run_sahi_inference():
    model_path = r"D:\zycgit\ZDevelop_Confocal\xxp_ui\deploy\weights\best.torchscript"
    image_path = r"D:\zycgit\ZDevelop_Confocal\xxp_ui\ultralytics_gcy\assets\bus.jpg"

    print("=== ENV CHECK ===")
    print("ultralytics:", ultralytics.__version__)
    print("sahi:", sahi.__version__)
    print("torch:", torch.__version__)
    print("cuda_available:", torch.cuda.is_available())
    print("model_path:", model_path, "exists:", os.path.exists(model_path))
    print("image_path:", image_path, "exists:", os.path.exists(image_path))
    assert os.path.exists(model_path), "model_path not found"
    assert os.path.exists(image_path), "image_path not found"

    # 1) load yolo model
    yolo_model = YOLO(model_path, task="detect")
    print("YOLO object:", type(yolo_model))
    print("YOLO internal model:", type(getattr(yolo_model, "model", None)))

    # 2) 强制验证 torchscript 是否支持 .to()（如果这里失败，你工程里SAHI必挂）
    try:
        yolo_model.to("cuda:0" if torch.cuda.is_available() else "cpu")
        print("[CHECK] yolo_model.to(...) OK")
    except Exception as e:
        print("[CHECK] yolo_model.to(...) FAILED:", repr(e))

    # 3) build SAHI detection model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model=yolo_model,
        confidence_threshold=0.25,
        # 不传 device
    )

    # 4) sliced prediction
    result = get_sliced_prediction(
        image=image_path,
        detection_model=detection_model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2
    )

    # 5) visualize
    img_bgr = cv2.imread(image_path)
    output_dict = visualize_object_predictions(
        image=img_bgr,
        object_prediction_list=result.object_prediction_list,
    )

    out_path = r"D:\zycgit\ZDevelop_Confocal\xxp_ui\results\sahi_test_out.jpg"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, output_dict["image"])
    print("[OK] saved:", out_path, "detections:", len(result.object_prediction_list))

    # 可选：本地有桌面再开窗口
    try:
        cv2.imshow("SAHI Fixed Result", output_dict["image"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("[WARN] imshow failed (maybe no GUI):", repr(e))

if __name__ == "__main__":
    run_sahi_inference()
