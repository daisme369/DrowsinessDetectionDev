from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drowsiness_detection.models import build_geometric_classifier, save_geometric_classifier
from drowsiness_detection.training import compute_classification_metrics, feature_matrix_from_rows
from drowsiness_detection.utils import load_baseline_b_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Baseline B frame-level geometric feature classifier.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline_b.yaml"))
    parser.add_argument("--features", default=None, help="CSV from scripts/extract_baseline_b_features.py.")
    parser.add_argument("--architecture", default=None, choices=["random_forest", "logistic_regression", "svm"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--report-dir", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_baseline_b_config(args.config)
    if args.architecture:
        config.model.architecture = args.architecture
    if args.report_dir:
        config.training.report_dir = args.report_dir

    feature_path = Path(args.features or config.training.feature_path)
    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature CSV does not exist: {feature_path}. "
            "Create it with `python scripts/extract_baseline_b_features.py`."
        )

    rows = _read_feature_rows(feature_path)
    train_rows = _rows_for_split(rows, config.dataset.train_split)
    val_rows = _rows_for_split(rows, config.dataset.val_split)
    test_rows = _rows_for_split(rows, config.dataset.test_split)
    feature_names = list(config.model.feature_names)

    x_train = feature_matrix_from_rows(train_rows, feature_names)
    y_train = _labels_from_rows(train_rows)
    model = build_geometric_classifier(
        config.model,
        use_class_weights=config.training.use_class_weights,
        seed=config.training.seed,
    )

    started = time.perf_counter()
    print(
        f"Training Baseline B frame classifier | architecture={config.model.architecture} "
        f"| train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}",
        flush=True,
    )
    model.fit(x_train, y_train)

    train_metrics = _evaluate_rows(model, train_rows, feature_names, config.dataset.class_names, config.inference.closed_class_index)
    val_metrics = _evaluate_rows(model, val_rows, feature_names, config.dataset.class_names, config.inference.closed_class_index)
    test_metrics = _evaluate_rows(model, test_rows, feature_names, config.dataset.class_names, config.inference.closed_class_index)

    model_path = Path(args.model_path or config.inference.model_path)
    metadata = {
        "baseline": "baseline_b_frame_geometric",
        "architecture": config.model.architecture,
        "feature_names": feature_names,
        "class_names": list(config.dataset.class_names),
        "closed_class_index": config.inference.closed_class_index,
        "config": args.config,
    }
    save_geometric_classifier(model, model_path, metadata)

    report_dir = Path(config.training.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "config": args.config,
        "feature_path": str(feature_path),
        "model_path": str(model_path),
        "model_bytes": model_path.stat().st_size,
        "architecture": config.model.architecture,
        "feature_names": feature_names,
        "class_names": list(config.dataset.class_names),
        "class_counts": {
            "train": dict(Counter(y_train)),
            "val": dict(Counter(_labels_from_rows(val_rows))),
            "test": dict(Counter(_labels_from_rows(test_rows))),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "train_metrics": train_metrics.to_dict(),
        "val_metrics": val_metrics.to_dict(),
        "test_metrics": test_metrics.to_dict(),
        "model_config": asdict(config.model),
    }
    report_path = report_dir / "train_metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"Saved model: {model_path}\n"
        f"Saved report: {report_path}\n"
        f"test_acc={test_metrics.accuracy:.4f} test_macro_f1={test_metrics.macro_f1:.4f} "
        f"test_pos_fnr={test_metrics.positive_false_negative_rate:.4f}",
        flush=True,
    )
    return 0


def _read_feature_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows_for_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise RuntimeError(f"No feature rows found for split: {split}")
    return selected


def _labels_from_rows(rows: list[dict[str, str]]) -> list[int]:
    return [int(row["label"]) for row in rows]


def _evaluate_rows(
    model: Any,
    rows: list[dict[str, str]],
    feature_names: list[str],
    class_names: list[str],
    positive_label: int,
) -> Any:
    x_values = feature_matrix_from_rows(rows, feature_names)
    y_true = _labels_from_rows(rows)
    y_pred = [int(value) for value in model.predict(x_values).tolist()]
    return compute_classification_metrics(
        y_true,
        y_pred,
        class_names=class_names,
        positive_label=positive_label,
    )


if __name__ == "__main__":
    raise SystemExit(main())
