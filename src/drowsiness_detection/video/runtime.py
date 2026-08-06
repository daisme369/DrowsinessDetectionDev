from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        return


def collect_environment() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "git_commit": git_commit(),
    }
    for package_name in ["cv2", "mediapipe", "numpy", "sklearn", "torch", "torchvision"]:
        payload[package_name] = _package_version(package_name)
    try:
        import torch

        payload["cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_version"] = torch.version.cuda
        payload["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    except Exception:
        payload["cuda_available"] = False
        payload["cuda_version"] = ""
        payload["gpu_name"] = ""
    return payload


def write_environment(path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(collect_environment(), indent=2), encoding="utf-8")


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def _package_version(package_name: str) -> str:
    try:
        module = __import__(package_name)
    except Exception:
        return ""
    return str(getattr(module, "__version__", ""))
