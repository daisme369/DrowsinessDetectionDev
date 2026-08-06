from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drowsiness_detection.video import load_video_config, save_resolved_config
from drowsiness_detection.video.manifest import discover_uta_rldd_videos, read_video_manifest, write_video_manifest
from drowsiness_detection.video.splits import create_five_fold_subject_splits, write_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Baseline D five-fold subject-wise evaluation.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "five_fold.yaml"))
    parser.add_argument("--experiment", default="e4")
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_video_config(args.config)
    print(json.dumps(config, indent=2), flush=True)
    manifest_path = Path(config["dataset"]["manifest_path"])
    if manifest_path.exists():
        records = read_video_manifest(manifest_path)
    else:
        records = discover_uta_rldd_videos(config, probe_video=True)
        write_video_manifest(records, manifest_path)

    split_dir = Path(config["split"].get("output_dir", "artifacts/splits"))
    folds = create_five_fold_subject_splits(
        records,
        number_of_folds=int(config["split"].get("number_of_folds", 5)),
        seed=int(config["split"].get("random_seed", config["project"].get("random_seed", 42))),
    )
    for index, split_manifest in enumerate(folds, start=1):
        write_split_manifest(split_manifest, split_dir / f"fold_{index}.json")

    cross_root = Path(config["project"]["output_dir"]) / "cross_validation"
    cross_root.mkdir(parents=True, exist_ok=True)
    metric_paths: list[Path] = []
    video_prediction_paths: list[Path] = []

    for fold_index in args.folds:
        if fold_index < 1 or fold_index > len(folds):
            raise ValueError(f"Fold index out of range: {fold_index}")
        fold_config_path = _write_fold_config(config, cross_root, fold_index, args.experiment)
        fold_split_path = split_dir / f"fold_{fold_index}.json"
        fold_stats_path = Path(config["project"]["output_dir"]) / "preprocessing" / f"fold_{fold_index}_behavior_stats.json"
        fold_output = cross_root / f"fold_{fold_index}"
        checkpoint_path = Path("checkpoints") / "baseline_d" / f"fold_{fold_index}" / "best_video_macro_f1.pt"

        commands: list[list[str]] = []
        if not args.skip_preprocess:
            preprocess_cmd = [
                sys.executable,
                str(ROOT / "scripts" / "preprocess.py"),
                "--config",
                str(fold_config_path),
                "--fold-index",
                str(fold_index),
            ]
            if args.max_videos > 0:
                preprocess_cmd.extend(["--max-videos", str(args.max_videos)])
            if args.max_frames_per_video > 0:
                preprocess_cmd.extend(["--max-frames-per-video", str(args.max_frames_per_video)])
            commands.append(preprocess_cmd)

        train_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--config",
            str(fold_config_path),
            "--split-manifest",
            str(fold_split_path),
            "--behavior-stats",
            str(fold_stats_path),
        ]
        if args.epochs > 0:
            train_cmd.extend(["--epochs", str(args.epochs)])
        commands.append(train_cmd)
        commands.append(
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate.py"),
                "--config",
                str(fold_config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--split",
                "test",
                "--behavior-stats",
                str(fold_stats_path),
                "--output-dir",
                str(fold_output),
            ]
        )

        for command in commands:
            print(" ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)
        metric_paths.append(fold_output / "metrics" / "test_metrics.json")
        video_prediction_paths.append(fold_output / "predictions" / "test_video_predictions.csv")

    if not args.dry_run:
        _write_cross_validation_summary(metric_paths, video_prediction_paths, cross_root)
    return 0


def _write_fold_config(base_config: dict, cross_root: Path, fold_index: int, experiment: str) -> Path:
    config = json.loads(json.dumps(base_config))
    config["experiment"]["id"] = experiment
    if experiment == "e4":
        config["experiment"]["name"] = "resnet18_ear_mar_lstm"
        config["experiment"]["use_visual"] = True
        config["experiment"]["use_behavior"] = True
        config["experiment"]["temporal_model"] = "lstm"
    config["training"]["checkpoint_dir"] = str(Path("checkpoints") / "baseline_d" / f"fold_{fold_index}")
    config["split"]["mode"] = "five_fold"
    output_path = cross_root / "configs" / f"fold_{fold_index}_{experiment}.yaml"
    save_resolved_config(config, output_path)
    return output_path


def _write_cross_validation_summary(metric_paths: list[Path], prediction_paths: list[Path], output_dir: Path) -> None:
    rows = []
    for fold_index, path in enumerate(metric_paths, start=1):
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        video = payload["video"]
        rows.append(
            {
                "fold": fold_index,
                "video_accuracy": float(video["accuracy"]),
                "video_balanced_accuracy": float(video["balanced_accuracy"]),
                "video_macro_f1": float(video["macro_f1"]),
                "video_weighted_f1": float(video["weighted_f1"]),
            }
        )
    summary = {"folds": rows, "mean": {}, "std": {}}
    for key in ["video_accuracy", "video_balanced_accuracy", "video_macro_f1", "video_weighted_f1"]:
        values = [row[key] for row in rows]
        if values:
            summary["mean"][key] = sum(values) / len(values)
            summary["std"][key] = _std(values)
    output_dir.joinpath("cross_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _combine_video_predictions(prediction_paths, output_dir / "out_of_fold_video_predictions.csv")


def _combine_video_predictions(paths: list[Path], output_path: Path) -> None:
    combined: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for fold_index, path in enumerate(paths, start=1):
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            for row in reader:
                row["fold"] = str(fold_index)
                combined.append(row)
    if fieldnames is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", *fieldnames])
        writer.writeheader()
        writer.writerows(combined)


def _std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


if __name__ == "__main__":
    raise SystemExit(main())
