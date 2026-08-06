from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drowsiness_detection.video import load_video_config
from drowsiness_detection.video.manifest import read_video_manifest
from drowsiness_detection.video.splits import create_five_fold_subject_splits, create_fixed_subject_split, write_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build subject-wise UTA-RLDD splits for Baseline D.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "development.yaml"))
    parser.add_argument("--manifest", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_video_config(args.config)
    print(json.dumps(config, indent=2), flush=True)
    records = read_video_manifest(args.manifest or config["dataset"]["manifest_path"])
    split_config = config["split"]
    output_dir = Path(split_config.get("output_dir", "artifacts/splits"))
    mode = split_config.get("mode", "fixed")
    seed = int(split_config.get("random_seed", config["project"].get("random_seed", 42)))

    if mode == "fixed":
        manifest = create_fixed_subject_split(
            records,
            train_ratio=float(split_config["train_ratio"]),
            validation_ratio=float(split_config["validation_ratio"]),
            test_ratio=float(split_config["test_ratio"]),
            seed=seed,
        )
        output = output_dir / f"fixed_split_seed_{seed}.json"
        write_split_manifest(manifest, output)
        print(f"Saved fixed split: {output}", flush=True)
        print(
            f"subjects train={len(manifest.train_subjects)} validation={len(manifest.validation_subjects)} "
            f"test={len(manifest.test_subjects)}",
            flush=True,
        )
        return 0

    if mode in {"five_fold", "cross_validation"}:
        folds = create_five_fold_subject_splits(
            records,
            number_of_folds=int(split_config["number_of_folds"]),
            seed=seed,
        )
        for index, manifest in enumerate(folds, start=1):
            output = output_dir / f"fold_{index}.json"
            write_split_manifest(manifest, output)
            print(f"Saved fold {index}: {output}", flush=True)
        return 0

    raise ValueError(f"Unsupported split mode: {mode}")


if __name__ == "__main__":
    raise SystemExit(main())
