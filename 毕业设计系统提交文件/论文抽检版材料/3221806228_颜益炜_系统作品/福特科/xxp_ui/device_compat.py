# -*- coding: utf-8 -*-
"""Runtime device selection for DAC-3D offline inference.

macOS does not provide NVIDIA CUDA. This module keeps accelerator-oriented call
sites portable by resolving the requested backend to CUDA, Apple MPS, or CPU.
MLX is detected for diagnostics, but the current SAHI/Ultralytics detector still
uses PyTorch devices. When PyTorch MPS is unavailable, the ONNX/CoreML detector
can still use Apple's CoreML runtime for exported ONNX models.
"""

from __future__ import annotations

from functools import lru_cache
import os
from importlib.util import find_spec
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = PROJECT_ROOT / "runtime"
for config_dir in (RUNTIME_DIR / "ultralytics_config", RUNTIME_DIR / "matplotlib"):
    config_dir.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("YOLO_CONFIG_DIR", str(RUNTIME_DIR / "ultralytics_config"))
os.environ.setdefault("MPLCONFIGDIR", str(RUNTIME_DIR / "matplotlib"))

import torch


@dataclass(frozen=True)
class RuntimeDevice:
    name: str
    label: str
    available: bool


@dataclass(frozen=True)
class BackendStatus:
    built: bool
    available: bool
    reason: str = ""


TRUTHY = {"1", "true", "yes", "on"}
FALSEY = {"0", "false", "no", "off"}


def _env_device_preference() -> str | None:
    for key in ("DAC3D_INFERENCE_DEVICE", "DAC3D_DEVICE"):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return None


@lru_cache(maxsize=1)
def mps_status() -> BackendStatus:
    backend = getattr(torch.backends, "mps", None)
    if not backend:
        return BackendStatus(False, False, "torch.backends.mps 不存在")
    if not backend.is_built():
        return BackendStatus(False, False, "当前 PyTorch 未编译 MPS 后端")
    if not backend.is_available():
        reason = "torch.backends.mps.is_available() 返回 False"
        try:
            torch.ones(1, device="mps")
        except Exception as exc:
            reason = f"{reason}: {exc}"
        return BackendStatus(True, False, reason)
    try:
        torch.ones(1, device="mps")
    except Exception as exc:
        return BackendStatus(True, False, f"MPS 张量创建失败: {exc}")
    return BackendStatus(True, True, "")


def mps_available() -> bool:
    return mps_status().available


def mlx_available() -> bool:
    try:
        return find_spec("mlx.core") is not None
    except ModuleNotFoundError:
        return False


@lru_cache(maxsize=1)
def coreml_status() -> BackendStatus:
    if find_spec("onnxruntime") is None:
        return BackendStatus(False, False, "onnxruntime 未安装")
    try:
        import onnxruntime as ort
    except Exception as exc:
        return BackendStatus(False, False, f"onnxruntime 导入失败: {exc}")

    providers = ort.get_available_providers()
    if "CoreMLExecutionProvider" not in providers:
        return BackendStatus(True, False, f"ONNX Runtime providers: {providers}")
    return BackendStatus(True, True, "")


def coreml_available() -> bool:
    return coreml_status().available


def mps_allowed() -> bool:
    """Return whether Apple MPS should be used for this run.

    The DAC-3D offline detector uses SAHI/Ultralytics slicing, which can hit
    Metal compiler failures in some headless/offscreen macOS sessions. MPS is
    enabled by default on Macs that expose it; set DAC3D_DISABLE_MPS=1 or
    DAC3D_ENABLE_MPS=0 to force CPU fallback.
    """

    disabled = os.getenv("DAC3D_DISABLE_MPS", "").strip().lower()
    if disabled in TRUTHY:
        return False
    enabled = os.getenv("DAC3D_ENABLE_MPS", "").strip().lower()
    if enabled in FALSEY:
        return False
    return True


def coreml_allowed() -> bool:
    disabled = os.getenv("DAC3D_DISABLE_COREML", "").strip().lower()
    if disabled in TRUTHY:
        return False
    enabled = os.getenv("DAC3D_ENABLE_COREML", "").strip().lower()
    if enabled in FALSEY:
        return False
    return True


def select_inference_device(preferred: str = "auto") -> RuntimeDevice:
    """Return the best available device for inference.

    Order for auto requests: NVIDIA CUDA -> Apple MPS -> ONNX/CoreML -> CPU.

    Set DAC3D_INFERENCE_DEVICE (or DAC3D_DEVICE) to one of auto, cuda, mps,
    coreml, mlx, or cpu to override call sites. MLX is not a PyTorch device, so
    "mlx" means "use Apple acceleration" and resolves to MPS or CoreML when
    the detector can use one of them.
    """

    requested = (_env_device_preference() or preferred or "auto").strip().lower()
    requested = requested.replace("metal", "mps")

    if requested in {"cpu", "none"}:
        return RuntimeDevice("cpu", "CPU", False)

    if requested in {"auto", "cuda", "cuda:0", "0", "gpu"}:
        if torch.cuda.is_available():
            return RuntimeDevice("cuda:0", torch.cuda.get_device_name(0), True)
        status = mps_status()
        if mps_allowed() and status.available:
            return RuntimeDevice("mps", "Apple Metal Performance Shaders", True)
        coreml = coreml_status()
        if coreml_allowed() and coreml.available:
            return RuntimeDevice("coreml", "ONNX Runtime CoreMLExecutionProvider", True)
        if status.built and status.reason:
            return RuntimeDevice("cpu", f"CPU (MPS unavailable: {status.reason})", False)
        return RuntimeDevice("cpu", "CPU", False)

    if requested in {"mps", "mps:0", "apple"}:
        status = mps_status()
        if mps_allowed() and status.available:
            return RuntimeDevice("mps", "Apple Metal Performance Shaders", True)
        reason = status.reason or "MPS unavailable"
        return RuntimeDevice("cpu", f"CPU (MPS requested but unavailable: {reason})", False)

    if requested in {"coreml", "onnx-coreml", "onnx_coreml"}:
        status = coreml_status()
        if coreml_allowed() and status.available:
            return RuntimeDevice("coreml", "ONNX Runtime CoreMLExecutionProvider", True)
        reason = status.reason or "CoreMLExecutionProvider unavailable"
        return RuntimeDevice("cpu", f"CPU (CoreML requested but unavailable: {reason})", False)

    if requested == "mlx":
        status = mps_status()
        if mps_allowed() and status.available:
            label = "Apple MPS (MLX requested; detector uses PyTorch)"
            return RuntimeDevice("mps", label, True)
        coreml = coreml_status()
        if coreml_allowed() and coreml.available:
            label = "ONNX Runtime CoreML (MLX requested; detector has no MLX backend)"
            return RuntimeDevice("coreml", label, True)
        reason = status.reason or "PyTorch MPS unavailable"
        label = f"CPU (MLX requested; {reason})"
        return RuntimeDevice("cpu", label, False)

    return RuntimeDevice("cpu", "CPU", False)


def select_torch_device(preferred: str = "auto") -> torch.device:
    device = select_inference_device(preferred).name
    if device == "coreml":
        return torch.device("cpu")
    return torch.device(device)


def get_device_usage(device_name: str) -> dict[str, float] | None:
    if not device_name.startswith("cuda"):
        return None
    try:
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return {
            "allocated": memory_allocated,
            "reserved": memory_reserved,
            "total": memory_total,
            "utilization": (memory_allocated / memory_total) * 100,
        }
    except Exception:
        return None
