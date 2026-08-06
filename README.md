# Drowsiness Detection System

This repository contains an edge-oriented driver drowsiness detection research prototype.

Implemented baselines:

- **Baseline A: OpenCV + facial landmarks + temporal rules**. It uses OpenCV for capture/display, MediaPipe Face Mesh for facial landmarks, geometric EAR/MAR/head-pose features, and a finite-state temporal rule system.
- **Baseline B: Frame-level geometric feature classifier**. It extracts MediaPipe landmark features such as EAR and MAR from the YawDD image dataset in `data/`, trains a lightweight sklearn classifier by frame, benchmarks latency, and runs a live/video demo without temporal training.

See `docs/context.md` for the research context, `docs/BaselineA.md` for Baseline A commands, and `docs/BaselineB.md` for Baseline B feature extraction, training, evaluation, benchmark, and run commands.
