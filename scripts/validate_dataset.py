from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drowsiness_detection.video import load_video_config
from drowsiness_detection.video.manifest import discover_uta_rldd_videos, read_video_manifest, write_video_manifest
from drowsiness_detection.video.validate import validate_video_records, write_validation_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate UTA-RLDD video manifest for Baseline D.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--report", default=str(ROOT / "artifacts" / "data_validation_report.json"))
    parser.add_argument("--invalid-output", default=str(ROOT / "artifacts" / "invalid_videos.csv"))
    parser.add_argument("--no-decode-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_video_config(args.config)
    print(json.dumps(config, indent=2), flush=True)
    manifest_path = Path(args.manifest or config["dataset"]["manifest_path"])
    if manifest_path.exists():
        records = read_video_manifest(manifest_path)
    else:
        records = discover_uta_rldd_videos(config, probe_video=True)
        write_video_manifest(records, manifest_path)

    report, invalid = validate_video_records(records, config, decode_one_frame=not args.no_decode_check)
    write_validation_outputs(
        report,
        invalid,
        report_path=args.report,
        invalid_path=args.invalid_output,
    )
    print(
        f"Validation complete: videos={report['video_count']} subjects={report['subject_count']} "
        f"invalid_records={report['invalid_video_count']}",
        flush=True,
    )
    print(f"Saved report: {args.report}", flush=True)
    print(f"Saved invalid video CSV: {args.invalid_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
