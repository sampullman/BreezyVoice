import os
from contextlib import nullcontext
from typing import Any, Iterable, List, Optional


def _get_int_env(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value in (None, ""):
        return None
    return int(value)


def _get_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_torch_runtime(torch_module: Any) -> None:
    num_threads = _get_int_env("BREEZYVOICE_NUM_THREADS")
    if num_threads is not None and num_threads > 0:
        torch_module.set_num_threads(num_threads)
    cudnn_backend = getattr(torch_module.backends, "cudnn", None)
    if cudnn_backend is not None:
        cudnn_backend.benchmark = _get_bool_env("BREEZYVOICE_TORCH_BENCHMARK", True)
        cudnn_backend.deterministic = _get_bool_env("BREEZYVOICE_TORCH_DETERMINISTIC", False)
    matmul_precision = os.environ.get("BREEZYVOICE_FLOAT32_MATMUL_PRECISION", "").strip().lower()
    if matmul_precision:
        torch_module.set_float32_matmul_precision(matmul_precision)


def get_torch_device(torch_module: Any):
    requested = os.environ.get("BREEZYVOICE_DEVICE", "auto").strip().lower()
    if requested in ("", "auto"):
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")
    if requested in ("cuda", "gpu", "rocm", "hip"):
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        raise RuntimeError(
            f"BREEZYVOICE_DEVICE={requested} was requested but torch.cuda.is_available() is False"
        )
    if requested == "mps":
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        raise RuntimeError("BREEZYVOICE_DEVICE=mps was requested but MPS is unavailable")
    if requested == "cpu":
        return torch_module.device("cpu")
    raise ValueError(f"Unsupported BREEZYVOICE_DEVICE value: {requested}")


def clear_device_cache(torch_module: Any, device: Any) -> None:
    if not _get_bool_env("BREEZYVOICE_EMPTY_CACHE", False):
        return
    if device.type == "cuda" and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
        return
    mps = getattr(torch_module, "mps", None)
    if device.type == "mps" and mps is not None and hasattr(mps, "empty_cache"):
        mps.empty_cache()


def describe_torch_runtime(torch_module: Any, device: Any) -> dict:
    summary = {
        "device": str(device),
        "num_threads": torch_module.get_num_threads(),
        "cuda_is_available": bool(torch_module.cuda.is_available()),
        "cuda_version": getattr(torch_module.version, "cuda", None),
        "hip_version": getattr(torch_module.version, "hip", None),
    }
    if hasattr(torch_module.backends, "mps"):
        summary["mps_is_available"] = bool(torch_module.backends.mps.is_available())
    if device.type == "cuda" and torch_module.cuda.is_available():
        try:
            summary["gpu_name"] = torch_module.cuda.get_device_name(device)
        except Exception:
            pass
    return summary


def should_log_stage_timings() -> bool:
    return _get_bool_env("BREEZYVOICE_LOG_STAGE_TIMINGS", True)


def _normalize_amp_dtype(dtype_name: str) -> Optional[str]:
    normalized = dtype_name.strip().lower()
    if normalized in ("", "off", "none", "false", "0", "disable", "disabled"):
        return None
    if normalized in ("fp16", "float16", "half"):
        return "float16"
    if normalized in ("bf16", "bfloat16"):
        return "bfloat16"
    raise ValueError(
        "Unsupported AMP dtype value "
        f"{dtype_name!r}; expected one of float16, bfloat16, or off"
    )


def get_amp_dtype_name(stage: str) -> Optional[str]:
    stage_key = stage.strip().upper()
    stage_value = os.environ.get(f"BREEZYVOICE_{stage_key}_AMP")
    if stage_value is not None:
        return _normalize_amp_dtype(stage_value)
    shared_value = os.environ.get("BREEZYVOICE_TTS_AMP")
    if shared_value is not None:
        return _normalize_amp_dtype(shared_value)
    return None


def get_amp_dtype(torch_module: Any, stage: str):
    dtype_name = get_amp_dtype_name(stage)
    if dtype_name is None:
        return None
    return getattr(torch_module, dtype_name)


def build_stage_autocast(torch_module: Any, device: Any, stage: str):
    dtype = get_amp_dtype(torch_module, stage)
    if dtype is None:
        return nullcontext()
    if device.type != "cuda":
        return nullcontext()
    return torch_module.autocast(device_type="cuda", dtype=dtype)


def build_ort_session_options(onnxruntime_module: Any):
    option = onnxruntime_module.SessionOptions()
    option.graph_optimization_level = onnxruntime_module.GraphOptimizationLevel.ORT_ENABLE_ALL

    intra_threads = _get_int_env("BREEZYVOICE_ORT_INTRA_OP_THREADS")
    if intra_threads is not None and intra_threads > 0:
        option.intra_op_num_threads = intra_threads

    inter_threads = _get_int_env("BREEZYVOICE_ORT_INTER_OP_THREADS")
    if inter_threads is not None and inter_threads >= 0:
        option.inter_op_num_threads = inter_threads

    return option


def select_ort_providers(
    onnxruntime_module: Any,
    *,
    preferred_device: Optional[Any] = None,
    allow_cpu_fallback: bool = True,
) -> List[str]:
    available = list(onnxruntime_module.get_available_providers())
    explicit = os.environ.get("BREEZYVOICE_ORT_PROVIDERS")
    if explicit:
        requested = [provider.strip() for provider in explicit.split(",") if provider.strip()]
        providers = [provider for provider in requested if provider in available]
        if providers or not allow_cpu_fallback:
            if providers:
                return providers
            raise RuntimeError(
                "None of the providers requested via BREEZYVOICE_ORT_PROVIDERS are available: "
                + ", ".join(requested)
            )
    candidates: List[str] = []
    requested_device = os.environ.get("BREEZYVOICE_ORT_DEVICE", "auto").strip().lower()
    if requested_device in ("", "auto") and preferred_device is not None:
        if preferred_device.type == "cuda":
            candidates = ["CUDAExecutionProvider", "ROCMExecutionProvider"]
        elif preferred_device.type == "mps":
            candidates = ["CoreMLExecutionProvider"]
    elif requested_device in ("cuda", "gpu"):
        candidates = ["CUDAExecutionProvider"]
    elif requested_device in ("rocm", "hip", "amd"):
        candidates = ["ROCMExecutionProvider"]
    elif requested_device == "mps":
        candidates = ["CoreMLExecutionProvider"]
    elif requested_device == "cpu":
        candidates = ["CPUExecutionProvider"]
    else:
        candidates = ["CUDAExecutionProvider", "ROCMExecutionProvider"]

    providers = [provider for provider in candidates if provider in available]
    if allow_cpu_fallback and "CPUExecutionProvider" in available and "CPUExecutionProvider" not in providers:
        providers.append("CPUExecutionProvider")
    if not providers:
        raise RuntimeError(
            "No matching ONNX Runtime providers are available. "
            f"Requested device={requested_device!r}, available={available!r}"
        )
    return providers


def describe_ort_runtime(onnxruntime_module: Any, providers: Iterable[str]) -> dict:
    return {
        "device": onnxruntime_module.get_device(),
        "available_providers": list(onnxruntime_module.get_available_providers()),
        "selected_providers": list(providers),
        "intra_op_threads": _get_int_env("BREEZYVOICE_ORT_INTRA_OP_THREADS"),
        "inter_op_threads": _get_int_env("BREEZYVOICE_ORT_INTER_OP_THREADS"),
    }
