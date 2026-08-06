from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drowsiness_detection.capture import OpenCVFrameSource
from drowsiness_detection.inference import BaselineAPipeline
from drowsiness_detection.utils import RollingPerformanceMeter, load_baseline_a_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Baseline A end-to-end runtime.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline_a.yaml"))
    parser.add_argument("--source", default=None, help="Camera index or video path. Overrides config.")
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    return parser.parse_args()


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, int(round((percent / 100.0) * (len(sorted_values) - 1))))
    return sorted_values[index]


def main() -> int:
    args = parse_args()
    config = load_baseline_a_config(args.config)
    if args.source is not None:
        config.camera.source = args.source

    source = OpenCVFrameSource(
        source=config.camera.source,
        width=config.camera.width,
        height=config.camera.height,
        target_fps=config.camera.target_fps,
    )
    meter = RollingPerformanceMeter(window_size=max(args.max_frames, 1))
    state_counts: Counter[str] = Counter()
    face_detected = 0
    latencies: list[float] = []

    with source, BaselineAPipeline(config) as pipeline:
        while len(latencies) < args.max_frames:
            captured = source.read()
            if captured is None:
                break
            result = pipeline.process_frame(captured.frame, captured.timestamp_seconds)
            meter.update(result.timestamp_seconds, result.inference_latency_ms)
            latencies.append(result.inference_latency_ms)
            state_counts[result.state.value] += 1
            face_detected += int(result.face_detected)

    snapshot = meter.snapshot()
    report = {
        "source": config.camera.source,
        "frames": len(latencies),
        "face_detection_success_rate": face_detected / len(latencies) if latencies else 0.0,
        "effective_fps": snapshot.effective_fps,
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 50),
            "p95": percentile(latencies, 95),
            "max": max(latencies) if latencies else 0.0,
        },
        "state_counts": dict(state_counts),
    }

    encoded = json.dumps(report, indent=2)
    print(encoded)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

