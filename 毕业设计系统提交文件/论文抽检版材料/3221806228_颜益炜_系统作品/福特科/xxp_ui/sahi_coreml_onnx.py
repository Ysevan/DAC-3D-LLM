# -*- coding: utf-8 -*-
"""SAHI-compatible ONNX Runtime/CoreML detector for DAC-3D.

This backend keeps SAHI slicing and post-processing intact while avoiding
PyTorch MPS tensor allocation. It feeds CPU numpy arrays to ONNX Runtime and
lets the CoreML execution provider accelerate supported ONNX subgraphs.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = PROJECT_ROOT / "runtime"
for config_dir in (RUNTIME_DIR / "ultralytics_config", RUNTIME_DIR / "matplotlib"):
    config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_DIR / "ultralytics_config"))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_DIR / "matplotlib"))

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression


class CoreMLOnnxDetectionModel(DetectionModel):
    required_packages = ["onnxruntime"]

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int | None = None,
        prefer_coreml: bool = True,
        **kwargs: Any,
    ):
        self.iou_threshold = iou_threshold
        self.prefer_coreml = prefer_coreml
        self.input_name = ""
        self.output_names: list[str] = []
        self.session_providers: list[str] = []
        self.stride = 32
        super().__init__(
            model_path=model_path,
            device="cpu",
            confidence_threshold=confidence_threshold,
            image_size=image_size,
            **kwargs,
        )

    def load_model(self):
        import onnxruntime as ort

        providers: list[Any] = ["CPUExecutionProvider"]
        if self.prefer_coreml and "CoreMLExecutionProvider" in ort.get_available_providers():
            providers = [
                (
                    "CoreMLExecutionProvider",
                    {
                        "ModelFormat": "MLProgram",
                        "MLComputeUnits": "ALL",
                        "RequireStaticInputShapes": "1",
                    },
                ),
                "CPUExecutionProvider",
            ]

        try:
            session = ort.InferenceSession(str(self.model_path), providers=providers)
        except Exception as exc:
            if providers == ["CPUExecutionProvider"]:
                raise
            print(f"[WARN] CoreML ONNX 初始化失败，回退 ONNX CPU: {exc}")
            session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])

        self.model = session
        self.session_providers = session.get_providers()
        self.input_name = session.get_inputs()[0].name
        self.output_names = [output.name for output in session.get_outputs()]
        self._apply_metadata(session.get_modelmeta().custom_metadata_map)

        input_shape = session.get_inputs()[0].shape
        if self.image_size is None and len(input_shape) == 4 and isinstance(input_shape[2], int):
            self.image_size = int(input_shape[2])

    def set_model(self, model: Any, **kwargs: Any):
        self.model = model

    def perform_inference(self, image: np.ndarray):
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")

        input_tensor = self._preprocess_rgb(image)
        outputs = self.model.run(self.output_names, {self.input_name: input_tensor})
        predictions = torch.from_numpy(outputs[0])
        predictions = non_max_suppression(
            predictions,
            conf_thres=self.confidence_threshold,
            iou_thres=self.iou_threshold,
            max_det=300,
        )[0]

        if len(predictions):
            predictions[:, :4] = ops.scale_boxes(input_tensor.shape[2:], predictions[:, :4], image.shape[:2]).round()

        self._original_predictions = [predictions.cpu()]
        self._original_shape = image.shape

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int]] | None = [[0, 0]],
        full_shape_list: list[list[int]] | None = None,
    ):
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)
        object_prediction_list_per_image = []

        for image_ind, image_predictions in enumerate(self._original_predictions or []):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []
            boxes = image_predictions.cpu().detach().numpy()

            for prediction in boxes:
                bbox = [float(max(0, coord)) for coord in prediction[:4].tolist()]
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])
                if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                    continue

                category_id = int(prediction[5])
                object_prediction_list.append(
                    ObjectPrediction(
                        bbox=bbox,
                        category_id=category_id,
                        category_name=self.category_mapping[str(category_id)],
                        score=float(prediction[4]),
                        shift_amount=shift_amount,
                        full_shape=self._original_shape[:2] if full_shape is None else full_shape,
                    )
                )

            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image

    def _preprocess_rgb(self, image: np.ndarray) -> np.ndarray:
        image_size = int(self.image_size or 640)
        letterbox = LetterBox(new_shape=(image_size, image_size), auto=False, stride=self.stride)
        resized = letterbox(image=image)
        tensor = resized.transpose((2, 0, 1))
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        tensor /= 255.0
        return tensor[None]

    def _apply_metadata(self, metadata: dict[str, str]):
        names = metadata.get("names") if metadata else None
        if names:
            parsed_names = ast.literal_eval(names)
            self.category_mapping = {str(key): value for key, value in parsed_names.items()}
        elif self.category_mapping is None:
            self.category_mapping = {"0": "defect"}

        imgsz = metadata.get("imgsz") if metadata else None
        if self.image_size is None and imgsz:
            parsed_imgsz = ast.literal_eval(imgsz)
            if isinstance(parsed_imgsz, (list, tuple)) and parsed_imgsz:
                self.image_size = int(parsed_imgsz[0])

        stride = metadata.get("stride") if metadata else None
        if stride:
            self.stride = int(stride)
