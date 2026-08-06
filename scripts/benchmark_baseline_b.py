from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drowsiness_detection.models import load_geometric_classifier
from drowsiness_detection.training import compute_classification_metrics, feature_matrix_from_rows
from drowsiness_detection.utils import load_baseline_b_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Baseline B frame-level geometric classifier.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline_b.yaml"))
    parser.add_argument("--features", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--split", default="test", choices=["train", "valid", "val", "test"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "benchmarks" / "baseline_b.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_baseline_b_config(args.config)
    feature_path = Path(args.features or config.training.feature_path)
    model_path = Path(args.model_path or config.inference.model_path)
    split = config.dataset.val_split if args.split == "val" else args.split

    rows = _rows_for_split(_read_feature_rows(feature_path), split)
    model, metadata = load_geometric_classifier(model_path)
    feature_names = list(metadata.get("feature_names") or config.model.feature_names)
    x_values = feature_matrix_from_rows(rows, feature_names)
    y_true = [int(row["label"]) for row in rows]

    latencies: list[float] = []
    y_pred: list[int] = []
    started = time.perf_counter()
    for index, row in enumerate(x_values):
        sample = row.reshape(1, -1)
        sample_started = time.perf_counter()
        prediction = int(model.predict(sample)[0])
        latency_ms = (time.perf_counter() - sample_started) * 1000.0
        if index >= args.warmup:
            latencies.append(latency_ms)
            y_pred.append(prediction)
    elapsed = time.perf_counter() - started
    measured_y_true = y_true[args.warmup :]
    metrics = compute_classification_metrics(
        measured_y_true,
        y_pred,
        class_names=config.dataset.class_names,
        positive_label=config.inference.closed_class_index,
    )

    report = {
        "model_path": str(model_path),
        "model_bytes": model_path.stat().st_size if model_path.exists() else 0,
        "feature_path": str(feature_path),
        "split": split,
        "sample_count": len(rows),
        "measured_samples": len(latencies),
        "elapsed_seconds": elapsed,
        "effective_fps": len(rows) / elapsed if elapsed > 0 else 0.0,
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "median": statistics.median(latencies) if latencies else 0.0,
            "p95": _percentile(latencies, 95.0),
            "min": min(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
        },
        "metrics": metrics.to_dict(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"samples={len(rows)} fps={report['effective_fps']:.1f} "
        f"latency_mean_ms={report['latency_ms']['mean']:.4f} macro_f1={metrics.macro_f1:.4f}\n"
        f"Saved benchmark: {output_path}",
        flush=True,
    )
    return 0


def _read_feature_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Feature CSV does not exist: {path}. "
            "Create it with `python scripts/extract_baseline_b_features.py`."
        )
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows_for_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise RuntimeError(f"No feature rows found for split: {split}")
    return selected


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return (ordered[lower] * (1.0 - fraction)) + (ordered[upper] * fraction)


if __name__ == "__main__":
    raise SystemExit(main())
