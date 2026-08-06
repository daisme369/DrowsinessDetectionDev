from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2

from drowsiness_detection.landmarks.face_landmarker import MediaPipeFaceMeshLandmarker
from drowsiness_detection.preprocessing.roi_extraction import extract_roi_from_yolo_box
from drowsiness_detection.training import build_frame_feature_dict, collect_yolo_roi_samples
from drowsiness_detection.utils import load_baseline_b_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frame-level geometric features for Baseline B.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline_b.yaml"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--splits", nargs="+", default=None, help="Dataset splits to process, e.g. train valid test.")
    parser.add_argument("--limit-samples", type=int, default=0, help="Limit each split for smoke tests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_baseline_b_config(args.config)
    output_path = Path(args.output or config.training.feature_path)
    report_path = Path(args.report or Path(config.training.report_dir) / "feature_extraction_report.json")
    splits = args.splits or [config.dataset.train_split, config.dataset.val_split, config.dataset.test_split]
    feature_names = list(config.model.feature_names)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    split_reports: dict[str, dict[str, int]] = {}
    started = time.perf_counter()

    landmarker = MediaPipeFaceMeshLandmarker(config.landmarks)
    try:
        for split in splits:
            samples = collect_yolo_roi_samples(config.dataset, split)
            if args.limit_samples > 0:
                samples = samples[: args.limit_samples]
            total = len(samples)
            detected = 0
            failed = 0
            for index, sample in enumerate(samples, start=1):
                image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
                if image is None:
                    failed += 1
                    continue

                roi, _roi_box = extract_roi_from_yolo_box(
                    image,
                    sample.yolo_box,
                    crop_mode=config.dataset.crop_mode,
                    padding=config.dataset.bbox_padding,
                    eye_band_y_min=config.dataset.eye_band_y_min,
                    eye_band_y_max=config.dataset.eye_band_y_max,
                    min_box_size=config.dataset.min_box_size,
                )
                detection_image = roi if roi is not None else image
                faces = landmarker.detect(detection_image)
                if not faces and roi is not None:
                    faces = landmarker.detect(image)
                if not faces:
                    failed += 1
                    continue

                try:
                    feature_dict = build_frame_feature_dict(faces[0])
                except ValueError:
                    failed += 1
                    continue

                row: dict[str, object] = {
                    "split": split,
                    "image_path": str(sample.image_path),
                    "label": sample.label,
                    "source_class_id": sample.source_class_id,
                }
                row.update({name: feature_dict[name] for name in feature_names})
                rows.append(row)
                detected += 1

                if index % 200 == 0:
                    print(f"{split}: {index}/{total} processed, detected={detected}, failed={failed}", flush=True)

            split_reports[split] = {
                "total": total,
                "detected": detected,
                "failed": failed,
            }
            print(f"{split}: total={total} detected={detected} failed={failed}", flush=True)
    finally:
        landmarker.close()

    fieldnames = ["split", "image_path", "label", "source_class_id", *feature_names]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "config": args.config,
        "output": str(output_path),
        "feature_names": feature_names,
        "row_count": len(rows),
        "splits": split_reports,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved features: {output_path} rows={len(rows)}", flush=True)
    print(f"Saved report: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
