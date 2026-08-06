from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drowsiness_detection.video import load_video_config
from drowsiness_detection.video.aggregation import write_clip_predictions, write_video_predictions
from drowsiness_detection.video.evaluation import (
    default_behavior_stats_path,
    default_clips_path,
    evaluate_model_on_loader,
    load_clip_dataset,
    make_dataloader,
)
from drowsiness_detection.video.metrics import DEFAULT_CLASS_NAMES
from drowsiness_detection.video.model import build_baseline_d_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Baseline D checkpoint.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "experiment_e4.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--clips", default=None)
    parser.add_argument("--behavior-stats", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_video_config(args.config)
    print(json.dumps(config, indent=2), flush=True)

    import torch
    from torch import nn

    checkpoint_path = Path(args.checkpoint or Path(config["training"]["checkpoint_dir"]) / "best_video_macro_f1.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    clips_path = Path(args.clips) if args.clips else default_clips_path(config)
    behavior_stats_path = Path(args.behavior_stats) if args.behavior_stats else default_behavior_stats_path(config)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = _torch_load(checkpoint_path, map_location=device)
    model = build_baseline_d_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    criterion = nn.CrossEntropyLoss()

    dataset = load_clip_dataset(
        config,
        split=args.split,
        clips_path=clips_path,
        behavior_stats_path=behavior_stats_path,
        training=False,
    )
    dataloader = make_dataloader(dataset, config, shuffle=False)
    result = evaluate_model_on_loader(model, dataloader, device, criterion)

    output_root = Path(args.output_dir or config["project"]["output_dir"])
    predictions_dir = output_root / "predictions"
    metrics_dir = output_root / "metrics"
    plots_dir = output_root / "plots"
    clip_path = predictions_dir / f"{args.split}_clip_predictions.csv"
    video_path = predictions_dir / f"{args.split}_video_predictions.csv"
    metrics_path = metrics_dir / f"{args.split}_metrics.json"
    write_clip_predictions(result["clip_predictions"], clip_path)
    write_video_predictions(result["video_predictions"], video_path)
    metrics_payload = {
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "clips": str(clips_path),
        "behavior_stats": str(behavior_stats_path),
        "loss": result["loss"],
        "clip": result["clip_metrics"].to_dict(),
        "video": result["video_metrics"].to_dict(),
    }
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    _write_confusion_matrix_png(
        result["video_metrics"].confusion_matrix,
        plots_dir / f"{args.split}_confusion_matrix.png",
    )
    print(f"Saved clip predictions: {clip_path}", flush=True)
    print(f"Saved video predictions: {video_path}", flush=True)
    print(f"Saved metrics: {metrics_path}", flush=True)
    print(f"Video macro F1: {result['video_metrics'].macro_f1:.4f}", flush=True)
    return 0


def _write_confusion_matrix_png(matrix: list[list[int]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5, 4), dpi=150)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(DEFAULT_CLASS_NAMES)), DEFAULT_CLASS_NAMES, rotation=30, ha="right")
    axis.set_yticks(range(len(DEFAULT_CLASS_NAMES)), DEFAULT_CLASS_NAMES)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, str(value), ha="center", va="center", color="black")
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _torch_load(path: str | Path, *, map_location):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


if __name__ == "__main__":
    raise SystemExit(main())
