from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "models" / "face_landmarker.task"
DEFAULT_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the MediaPipe Face Landmarker model.")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.url}")
    with urlopen(args.url, timeout=60) as response:
        data = response.read()

    output.write_bytes(data)
    print(f"Saved {len(data)} bytes to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
