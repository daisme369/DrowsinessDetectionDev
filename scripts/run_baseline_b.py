from __future__ import annotations

import argparse
from importlib import import_module
from importlib import metadata
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2

from drowsiness_detection.capture import OpenCVFrameSource
from drowsiness_detection.inference import BaselineBPipeline
from drowsiness_detection.landmarks.face_landmarker import resolve_model_asset_path
from drowsiness_detection.utils import RollingPerformanceMeter, load_baseline_b_config
from drowsiness_detection.visualization import draw_baseline_b_overlay


DEFAULT_IP_CAMERA_URL = "http://192.168.0.101:8080/video"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Baseline B frame-level geometric feature classifier.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline_b.yaml"))
    parser.add_argument("--source", default=DEFAULT_IP_CAMERA_URL, help="Camera index, video path, or IP camera stream URL.")
    parser.add_argument("--model-path", default=None, help="Override sklearn model path.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 means unlimited.")
    parser.add_argument("--no-display", action="store_true", help="Run headless and print periodic status.")
    parser.add_argument("--camera-only", action="store_true", help="Open the camera stream without MediaPipe/model.")
    parser.add_argument("--check-deps", action="store_true", help="Print dependency and artifact diagnostics.")
    return parser.parse_args()


def print_dependency_report(config_path: str) -> None:
    config = load_baseline_b_config(config_path)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"OpenCV: {cv2.__version__}")
    for package_name in ("scikit-learn", "mediapipe"):
        try:
            print(f"{package_name}: {metadata.version(package_name)}")
        except metadata.PackageNotFoundError:
            print(f"{package_name}: not installed")

    for module_name in ("mediapipe.tasks.python", "mediapipe.tasks.python.vision"):
        try:
            import_module(module_name)
            print(f"MediaPipe task import OK: {module_name}")
        except Exception as error:
            print(f"MediaPipe task import failed: {module_name}: {error}")

    face_model_path = resolve_model_asset_path(config.landmarks.model_asset_path)
    baseline_b_model_path = Path(config.inference.model_path)
    feature_path = Path(config.training.feature_path)
    print(f"Face Landmarker model: {face_model_path} | exists={face_model_path.exists()}")
    print(f"Baseline B feature CSV: {feature_path} | exists={feature_path.exists()}")
    print(f"Baseline B sklearn model: {baseline_b_model_path} | exists={baseline_b_model_path.exists()}")


def run_camera_only(args: argparse.Namespace, config_source: str | int) -> int:
    config = load_baseline_b_config(args.config)
    source = OpenCVFrameSource(
        source=config_source,
        width=config.camera.width,
        height=config.camera.height,
        target_fps=config.camera.target_fps,
    )
    with source:
        while True:
            captured = source.read()
            if captured is None:
                break
            if args.no_display:
                if captured.frame_index % 30 == 0:
                    height, width = captured.frame.shape[:2]
                    print(f"frame={captured.frame_index} size={width}x{height}", flush=True)
            else:
                cv2.imshow("IP Camera Test", captured.frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            if args.max_frames and captured.frame_index + 1 >= args.max_frames:
                break
    if not args.no_display:
        cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()
    if args.check_deps:
        print_dependency_report(args.config)
        return 0

    config = load_baseline_b_config(args.config)
    config.camera.source = args.source
    if args.model_path:
        config.inference.model_path = args.model_path
    if args.camera_only:
        return run_camera_only(args, config.camera.source)

    source = OpenCVFrameSource(
        source=config.camera.source,
        width=config.camera.width,
        height=config.camera.height,
        target_fps=config.camera.target_fps,
    )
    meter = RollingPerformanceMeter()
    try:
        with BaselineBPipeline(config) as pipeline, source:
            while True:
                captured = source.read()
                if captured is None:
                    break

                result = pipeline.process_frame(captured.frame, captured.timestamp_seconds)
                performance = meter.update(result.timestamp_seconds, result.inference_latency_ms)

                if args.no_display:
                    if captured.frame_index % 30 == 0:
                        print(
                            f"frame={captured.frame_index} "
                            f"state={result.state.value} "
                            f"face={result.face_detected} "
                            f"ear={_metric_value(result.features, 'mean_ear')} "
                            f"mar={_metric_value(result.features, 'mar')} "
                            f"pred={_prediction_value(result)} "
                            f"fps={performance.effective_fps:.1f} "
                            f"latency_ms={performance.last_latency_ms:.1f}",
                            flush=True,
                        )
                else:
                    display = draw_baseline_b_overlay(
                        captured.frame,
                        result,
                        performance,
                        show_metrics=config.display.show_metrics,
                    )
                    cv2.imshow(config.display.window_name, display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break

                if args.max_frames and captured.frame_index + 1 >= args.max_frames:
                    break
    finally:
        source.close()
        if not args.no_display:
            cv2.destroyAllWindows()
    return 0


def handle_runtime_error(error: RuntimeError | FileNotFoundError) -> int:
    print(str(error), file=sys.stderr)
    print("", file=sys.stderr)
    print("Extract Baseline B features:", file=sys.stderr)
    print("  python scripts/extract_baseline_b_features.py", file=sys.stderr)
    print("", file=sys.stderr)
    print("Train Baseline B:", file=sys.stderr)
    print("  python scripts/train_baseline_b.py", file=sys.stderr)
    print("", file=sys.stderr)
    print("Download Face Landmarker model:", file=sys.stderr)
    print("  python scripts/download_face_landmarker_model.py", file=sys.stderr)
    print("", file=sys.stderr)
    print("Camera-only test:", file=sys.stderr)
    print("  python scripts/run_baseline_b.py --camera-only", file=sys.stderr)
    print("", file=sys.stderr)
    print("Dependency check:", file=sys.stderr)
    print("  python scripts/run_baseline_b.py --check-deps", file=sys.stderr)
    return 2


def _metric_value(features: dict[str, float] | None, name: str) -> str:
    if not features or name not in features:
        return "n/a"
    return f"{features[name]:.3f}"


def _prediction_value(result: object) -> str:
    prediction = getattr(result, "prediction", None)
    if prediction is None:
        return "n/a"
    return f"{prediction.class_index}:{prediction.confidence:.2f}"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, FileNotFoundError) as error:
        raise SystemExit(handle_runtime_error(error))
