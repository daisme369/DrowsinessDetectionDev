from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drowsiness_detection.models import load_geometric_classifier
from drowsiness_detection.training import compute_classification_metrics, feature_matrix_from_rows
from drowsiness_detection.utils import load_baseline_b_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Baseline B frame-level geometric feature classifier.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline_b.yaml"))
    parser.add_argument("--features", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--split", default="test", choices=["train", "valid", "val", "test"])
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "reports" / "baseline_b" / "eval_metrics.json"))
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
    y_pred = [int(value) for value in model.predict(x_values).tolist()]
    metrics = compute_classification_metrics(
        y_true,
        y_pred,
        class_names=config.dataset.class_names,
        positive_label=config.inference.closed_class_index,
    )

    report = {
        "model_path": str(model_path),
        "feature_path": str(feature_path),
        "split": split,
        "sample_count": len(rows),
        "feature_names": feature_names,
        "metrics": metrics.to_dict(),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"split={split} samples={len(rows)} "
        f"acc={metrics.accuracy:.4f} macro_f1={metrics.macro_f1:.4f} "
        f"positive_fnr={metrics.positive_false_negative_rate:.4f}\n"
        f"Saved report: {output_path}",
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


if __name__ == "__main__":
    raise SystemExit(main())
