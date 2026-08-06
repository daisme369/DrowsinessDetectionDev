from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from drowsiness_detection.video import load_video_config, save_resolved_config
from drowsiness_detection.video.clips import build_clip_records, write_clip_manifest, write_rejected_clips
from drowsiness_detection.video.manifest import discover_uta_rldd_videos, read_video_manifest, write_video_manifest
from drowsiness_detection.video.preprocess import preprocess_videos, write_frame_manifest, write_processed_video_summary
from drowsiness_detection.video.splits import (
    create_five_fold_subject_splits,
    create_fixed_subject_split,
    read_split_manifest,
    write_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess UTA-RLDD videos for Baseline D.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "development.yaml"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--fold-index", type=int, default=1)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-frames-per-video", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_video_config(args.config)
    print(json.dumps(config, indent=2), flush=True)
    processed_dir = Path(config["dataset"]["processed_dir"])
    manifest_dir = processed_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, manifest_dir / "config_resolved.yaml")

    manifest_path = Path(args.manifest or config["dataset"]["manifest_path"])
    if manifest_path.exists():
        records = read_video_manifest(manifest_path)
    else:
        records = discover_uta_rldd_videos(config, probe_video=True)
        write_video_manifest(records, manifest_path)

    split_manifest_path = _ensure_split_manifest(config, records, args)
    split_manifest = read_split_manifest(split_manifest_path)
    print(f"Active split manifest: {split_manifest_path}", flush=True)

    summaries, frame_rows = preprocess_videos(
        records,
        config,
        max_videos=args.max_videos,
        max_frames_per_video=args.max_frames_per_video,
    )
    write_frame_manifest(frame_rows, manifest_dir / "frames.csv")
    write_processed_video_summary(summaries, manifest_dir / "processed_videos.csv")

    clips, rejected = build_clip_records(summaries, split_manifest, config)
    write_clip_manifest(clips, manifest_dir / "clips.csv")
    write_rejected_clips(rejected, Path(config["project"]["output_dir"]) / "preprocessing" / "rejected_clips.csv")
    stats = _behavior_stats_for_train_videos(summaries, set(split_manifest.train_video_ids))
    stats_path = _behavior_stats_path(config, args.fold_index)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Saved frames: {manifest_dir / 'frames.csv'} rows={len(frame_rows)}", flush=True)
    print(f"Saved clips: {manifest_dir / 'clips.csv'} clips={len(clips)} rejected={len(rejected)}", flush=True)
    print(f"Saved behavior stats: {stats_path}", flush=True)
    return 0


def _ensure_split_manifest(config: dict, records: list, args: argparse.Namespace) -> Path:
    if args.split_manifest:
        return Path(args.split_manifest)
    split_config = config["split"]
    output_dir = Path(split_config.get("output_dir", "artifacts/splits"))
    seed = int(split_config.get("random_seed", config["project"].get("random_seed", 42)))
    mode = split_config.get("mode", "fixed")
    if mode == "fixed":
        output = output_dir / f"fixed_split_seed_{seed}.json"
        if not output.exists():
            manifest = create_fixed_subject_split(
                records,
                train_ratio=float(split_config["train_ratio"]),
                validation_ratio=float(split_config["validation_ratio"]),
                test_ratio=float(split_config["test_ratio"]),
                seed=seed,
            )
            write_split_manifest(manifest, output)
        return output
    if mode in {"five_fold", "cross_validation"}:
        output = output_dir / f"fold_{args.fold_index}.json"
        if not output.exists():
            folds = create_five_fold_subject_splits(
                records,
                number_of_folds=int(split_config["number_of_folds"]),
                seed=seed,
            )
            for index, manifest in enumerate(folds, start=1):
                write_split_manifest(manifest, output_dir / f"fold_{index}.json")
        return output
    raise ValueError(f"Unsupported split mode: {mode}")


def _behavior_stats_for_train_videos(summaries: list, train_video_ids: set[str]) -> dict[str, float | int]:
    ears: list[np.ndarray] = []
    mars: list[np.ndarray] = []
    for summary in summaries:
        if summary.video_id not in train_video_ids:
            continue
        data = np.load(summary.npz_path)
        valid = data["valid_mask"].astype(bool)
        ears.append(data["ear"][valid].astype(np.float32))
        mars.append(data["mar"][valid].astype(np.float32))
    ear_values = np.concatenate(ears) if ears else np.asarray([0.0], dtype=np.float32)
    mar_values = np.concatenate(mars) if mars else np.asarray([0.0], dtype=np.float32)
    return {
        "ear_mean": float(ear_values.mean()),
        "ear_std": float(max(ear_values.std(), 1e-6)),
        "mar_mean": float(mar_values.mean()),
        "mar_std": float(max(mar_values.std(), 1e-6)),
        "train_frame_count": int(len(ear_values)),
    }


def _behavior_stats_path(config: dict, fold_index: int) -> Path:
    preprocessing_dir = Path(config["project"]["output_dir"]) / "preprocessing"
    split_mode = str(config.get("split", {}).get("mode", "fixed"))
    if split_mode in {"five_fold", "cross_validation"}:
        return preprocessing_dir / f"fold_{fold_index}_behavior_stats.json"
    return preprocessing_dir / "fixed_behavior_stats.json"


if __name__ == "__main__":
    raise SystemExit(main())
