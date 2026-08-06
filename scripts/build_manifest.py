from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drowsiness_detection.video import load_video_config
from drowsiness_detection.video.manifest import discover_uta_rldd_videos, write_video_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build UTA-RLDD video manifest for Baseline D.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-probe", action="store_true", help="Skip OpenCV metadata probing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_video_config(args.config)
    print(json.dumps(config, indent=2), flush=True)
    records = discover_uta_rldd_videos(config, probe_video=not args.no_probe)
    output_path = Path(args.output or config["dataset"]["manifest_path"])
    write_video_manifest(records, output_path)
    print(f"Saved manifest: {output_path} videos={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
