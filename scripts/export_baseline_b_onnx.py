from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from drowsiness_detection.models import build_eye_classifier
from drowsiness_detection.models.model_factory import load_checkpoint_state
from drowsiness_detection.utils import load_baseline_b_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Baseline B classifier checkpoint to ONNX.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline_b.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_baseline_b_config(args.config)
    if config.inference.backend == "sklearn" or config.model.architecture in {"random_forest", "logistic_regression", "svm"}:
        print(
            "Baseline B now defaults to a frame-level sklearn geometric feature classifier. "
            "This ONNX export script is only for the legacy pixel-CNN experiment. "
            "Use `python scripts/train_baseline_b.py` to create "
            "`artifacts/models/baseline_b/geometric_feature_classifier.pkl`.",
            flush=True,
        )
        return 2
    checkpoint_path = args.checkpoint or config.inference.checkpoint_path
    output_path = Path(args.output or config.inference.onnx_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = build_eye_classifier(config.model).to(device)
    load_checkpoint_state(model, checkpoint_path, map_location=device)
    model.eval()

    channels = 1 if config.model.grayscale else 3
    height, width = int(config.model.input_size[0]), int(config.model.input_size[1])
    dummy = torch.randn(1, channels, height, width, device=device)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        dynamo=False,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Exported ONNX: {output_path} ({output_path.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
