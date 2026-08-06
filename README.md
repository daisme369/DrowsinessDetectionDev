# Drowsiness Detection System

Driver drowsiness detection research prototype for comparing lightweight, explainable, and temporal approaches under edge-device constraints.

The project detects visual signs of drowsiness from a driver-facing camera or recorded video. Drowsiness detection is not a single-frame problem: a closed eye in one frame may be a normal blink, while prolonged eye closure, repeated yawning, abnormal head pitch, or high eye-closure ratio over time can indicate fatigue. This repository therefore separates frame capture, face/landmark processing, feature extraction, model inference, temporal aggregation, evaluation, and visualization.

## Approaches

### 1. Rule-Based Landmark Pipeline

Implemented as Baseline A.

```text
camera or video
    -> OpenCV frame capture
    -> MediaPipe face landmarks
    -> EAR, MAR, and head-pose features
    -> temporal state machine
    -> alert / possibly drowsy / drowsy / recovery
```

This is the most explainable approach. It uses hand-tuned thresholds for:

- Eye Aspect Ratio (EAR), for open versus closed eyes.
- Mouth Aspect Ratio (MAR), for yawning evidence.
- Head pitch, for nodding or abnormal head posture.
- PERCLOS-like windows, duration counters, and hysteresis, so the system does not raise warnings from a single frame.

Baseline A does not need a training dataset. It is mainly calibrated and tested on live camera streams or videos.

### 2. Frame-Only Geometric Classifier

Implemented as Baseline B.

```text
image
    -> YOLO face box from dataset annotation
    -> MediaPipe face landmarks
    -> EAR / MAR / head-pose feature vector
    -> scikit-learn classifier
    -> per-frame open_or_alert / closed_or_drowsy prediction
```

This approach learns a frame-level decision boundary from landmark-derived geometry instead of using only manual thresholds. It intentionally avoids temporal features such as blink duration, yawn duration, consecutive-frame counts, or PERCLOS during training because the dataset used here is image-based.

Baseline B uses the Roboflow YawDD export in `data/`:

- `data/train/images`, `data/train/labels`
- `data/valid/images`, `data/valid/labels`
- `data/test/images`, `data/test/labels`
- `data/data.yaml`

The current dataset is `yawdd` version 1 from Roboflow Universe, licensed CC BY 4.0. It contains 2160 resized images with YOLO-format face-box labels and two numeric class names: `0` and `1`. The project maps these to:

- `open_or_alert`
- `closed_or_drowsy`

### 3. Temporal Video Model

Implemented by the Baseline D video pipeline.

```text
UTA-RLDD video
    -> subject-wise manifest and split
    -> frame sampling
    -> face detection, alignment, and crop caching
    -> ResNet18 visual embedding per frame
    -> EAR + MAR behavior features
    -> LSTM temporal encoder
    -> clip-level and video-level classification
```

This approach treats drowsiness as a temporal state over a sequence of sampled frames. The main model combines visual embeddings from ResNet18 with behavior features (EAR and MAR), then classifies a temporal clip with an LSTM.

The temporal pipeline uses UTA-RLDD videos under:

```text
data/video/
```

The configured canonical labels are:

| Raw label | Canonical label | Class id |
| --- | --- | --- |
| `0` | `alert` | `0` |
| `5` | `low_vigilant` | `1` |
| `10` | `drowsy` | `2` |

The default manifest path is `data/manifests/videos.csv`, and processed clips are cached under `data/processed/baseline_d/`.

## Project Structure

```text
.
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- configs/
|-- data/
|-- docs/
|-- scripts/
|-- src/
|   `-- drowsiness_detection/
|       |-- capture/
|       |-- inference/
|       |-- landmarks/
|       |-- models/
|       |-- preprocessing/
|       |-- temporal/
|       |-- training/
|       |-- utils/
|       |-- video/
|       `-- visualization/
|-- tests/
|-- artifacts/
`-- checkpoints/
```

Main folders:

| Path | Description |
| --- | --- |
| `configs/` | YAML configuration files for Baseline A, Baseline B, development temporal runs, five-fold runs, and experiments. |
| `data/` | Local datasets. Contains the YawDD Roboflow image export and UTA-RLDD video/manifests/processed data. Usually not committed to Git. |
| `docs/` | Longer research notes and implementation details for the baselines. |
| `scripts/` | Command-line entry points for feature extraction, training, evaluation, benchmarking, preprocessing, and live/video demos. |
| `src/drowsiness_detection/capture/` | OpenCV camera, video, and stream input handling. |
| `src/drowsiness_detection/landmarks/` | MediaPipe face landmarker adapter, EAR/MAR geometry, and head-pose estimation. |
| `src/drowsiness_detection/preprocessing/` | ROI and face-box preprocessing utilities. |
| `src/drowsiness_detection/temporal/` | Rule-based temporal state machine and eye-state aggregation. |
| `src/drowsiness_detection/inference/` | Runtime pipelines and inference backends for Baseline A and Baseline B. |
| `src/drowsiness_detection/models/` | scikit-learn geometric classifier utilities and optional PyTorch model definitions. |
| `src/drowsiness_detection/training/` | Dataset loading, feature-vector building, and classification metrics for frame-level training. |
| `src/drowsiness_detection/video/` | UTA-RLDD manifest, split, preprocessing, temporal dataset, model, runtime, evaluation, and metrics code. |
| `src/drowsiness_detection/visualization/` | OpenCV overlays for live demos. |
| `tests/` | Pytest unit tests for config loading, geometry, temporal logic, splits, metrics, ROI extraction, and video helpers. |
| `artifacts/` | Generated reports, features, plots, benchmarks, explainability outputs, and downloaded landmarker/model files. |
| `checkpoints/` | Generated temporal-model checkpoints. |

## Requirements

Recommended Python version:

- Python 3.10 to 3.12 for full MediaPipe support.
- Python 3.13 may work for non-MediaPipe parts, but MediaPipe support is limited.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

For editable development with tests:

```bash
python -m pip install -e ".[dev]"
```

Download the MediaPipe Face Landmarker model used by the landmark pipelines:

```bash
python scripts/download_face_landmarker_model.py
```

The expected output file is:

```text
artifacts/models/face_landmarker.task
```

## How to Run

### Check Setup

Run dependency and artifact checks:

```bash
python scripts/run_baseline_a.py --check-deps
python scripts/run_baseline_b.py --check-deps
```

Run the test suite:

```bash
python -m pytest
```

### Run Approach 1: Rule-Based Landmark Pipeline

Use a local webcam:

```bash
python scripts/run_baseline_a.py --source 0
```

Use a video file:

```bash
python scripts/run_baseline_a.py --source path/to/video.mp4
```

Run headless for a short smoke test:

```bash
python scripts/run_baseline_a.py --source path/to/video.mp4 --no-display --max-frames 300
```

Benchmark end-to-end runtime:

```bash
python scripts/benchmark_baseline_a.py --source path/to/video.mp4 --max-frames 300 --output artifacts/benchmarks/baseline_a.json
```

### Run Approach 2: Frame-Only Geometric Classifier

Extract landmark-based frame features from the YawDD Roboflow dataset:

```bash
python scripts/extract_baseline_b_features.py
```

Train the default Random Forest classifier:

```bash
python scripts/train_baseline_b.py
```

Optional model choices:

```bash
python scripts/train_baseline_b.py --architecture logistic_regression
python scripts/train_baseline_b.py --architecture svm
```

Evaluate on the test split:

```bash
python scripts/evaluate_baseline_b.py --split test
```

Benchmark classifier latency:

```bash
python scripts/benchmark_baseline_b.py --split test
```

Run the live/video demo after training:

```bash
python scripts/run_baseline_b.py --source 0
```

Headless smoke test:

```bash
python scripts/run_baseline_b.py --source path/to/video.mp4 --no-display --max-frames 300
```

### Run Approach 3: Temporal Video Model

Place UTA-RLDD videos under `data/video/`. The discovery code expects subject folders and video filenames whose stems begin with the raw label values `0`, `5`, or `10`.

Build the video manifest:

```bash
python scripts/build_manifest.py --config configs/base.yaml
```

Validate the manifest and videos:

```bash
python scripts/validate_dataset.py --config configs/base.yaml
```

Preprocess videos into cached faces, behavior features, split manifests, and clip manifests:

```bash
python scripts/preprocess.py --config configs/development.yaml
```

For a quick smoke run:

```bash
python scripts/preprocess.py --config configs/development.yaml --max-videos 3 --max-frames-per-video 120
```

Train the hybrid ResNet18 + EAR/MAR + LSTM model:

```bash
python scripts/train.py --config configs/experiment_e4.yaml
```

For a quick training smoke run:

```bash
python scripts/train.py --config configs/experiment_e4.yaml --epochs 1 --max-train-clips 8 --max-validation-clips 4
```

Evaluate a checkpoint:

```bash
python scripts/evaluate.py --config configs/experiment_e4.yaml --checkpoint checkpoints/baseline_d/best_video_macro_f1.pt
```

Run five-fold subject-wise evaluation:

```bash
python scripts/run_cross_validation.py --config configs/five_fold.yaml --experiment e4
```

Generate explainability artifacts:

```bash
python scripts/explain.py --config configs/experiment_e4.yaml --checkpoint checkpoints/baseline_d/best_video_macro_f1.pt
```

## Useful Outputs

| Output | Created by |
| --- | --- |
| `artifacts/features/baseline_b_geometric_features.csv` | `scripts/extract_baseline_b_features.py` |
| `artifacts/models/baseline_b/geometric_feature_classifier.pkl` | `scripts/train_baseline_b.py` |
| `artifacts/reports/baseline_b/` | Baseline B training and evaluation |
| `artifacts/benchmarks/` | Benchmark scripts |
| `data/manifests/videos.csv` | `scripts/build_manifest.py` |
| `data/processed/baseline_d/` | `scripts/preprocess.py` |
| `artifacts/splits/` | Temporal split generation |
| `checkpoints/baseline_d/` | `scripts/train.py` |

## Notes

- Baseline A is rule-based and explainable, but thresholds must be calibrated for the target camera and lighting conditions.
- Baseline B is frame-only because the YawDD Roboflow export does not provide reliable temporal order.
- The temporal video model should be evaluated with subject-wise splits to avoid data leakage between train, validation, and test videos.
- `artifacts/`, `checkpoints/`, Python cache files, and local datasets can become large and are normally excluded from source control.
