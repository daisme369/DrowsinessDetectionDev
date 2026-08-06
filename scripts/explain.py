from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np

from drowsiness_detection.video import load_video_config
from drowsiness_detection.video.evaluation import default_behavior_stats_path, default_clips_path, load_clip_dataset, make_dataloader
from drowsiness_detection.video.model import build_baseline_d_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Baseline D explainability artifacts.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "experiment_e4.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--clips", default=None)
    parser.add_argument("--behavior-stats", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-clips", type=int, default=5)
    parser.add_argument("--frame-step", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-gradcam", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_video_config(args.config)
    print(json.dumps(config, indent=2), flush=True)

    import torch

    checkpoint_path = Path(args.checkpoint or Path(config["training"]["checkpoint_dir"]) / "best_video_macro_f1.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    clips_path = Path(args.clips) if args.clips else default_clips_path(config)
    behavior_stats_path = Path(args.behavior_stats) if args.behavior_stats else default_behavior_stats_path(config)
    output_dir = Path(args.output_dir or Path(config["project"]["output_dir"]) / "explainability")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = _torch_load(checkpoint_path, map_location=device)
    model = build_baseline_d_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = load_clip_dataset(
        config,
        split=args.split,
        clips_path=clips_path,
        behavior_stats_path=behavior_stats_path,
        training=False,
    )
    selected_count = min(int(args.max_clips), len(dataset))
    summary_rows: list[dict[str, object]] = []
    occlusion_rows: list[dict[str, object]] = []
    temporal_rows: list[dict[str, object]] = []

    for index in range(selected_count):
        item = dataset[index]
        clip = dataset.clips[index]
        frames = item["face_frames"].unsqueeze(0).to(device)
        behavior = item["behavior"].unsqueeze(0).to(device)
        valid_mask = item["valid_mask"].unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(frames, behavior, valid_mask)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        predicted_label = int(np.argmax(probabilities))
        summary_rows.append(
            {
                "clip_id": clip.clip_id,
                "video_id": clip.video_id,
                "subject_id": clip.subject_id,
                "true_label": int(clip.label_id),
                "predicted_label": predicted_label,
                "p_alert": float(probabilities[0]),
                "p_low_vigilant": float(probabilities[1]),
                "p_drowsy": float(probabilities[2]),
                "average_normalized_ear": float(item["behavior"][:, 0].numpy().mean()),
                "average_normalized_mar": float(item["behavior"][:, 1].numpy().mean()),
                "valid_frame_ratio": float(item["valid_mask"].numpy().mean()),
            }
        )
        temporal_rows.extend(_temporal_occlusion_rows(model, item, clip, device, predicted_label, probabilities[predicted_label]))
        occlusion_rows.extend(_region_occlusion_rows(model, item, clip, config, device, predicted_label, probabilities))
        _write_behavior_series(clip, output_dir / "temporal_series")
        if not args.skip_gradcam:
            _write_gradcam_overlays(model, item, clip, config, device, predicted_label, output_dir / "gradcam", args.frame_step)

    _write_rows(summary_rows, output_dir / "selected_clips.csv")
    _write_rows(temporal_rows, output_dir / "temporal_occlusion.csv")
    _write_rows(occlusion_rows, output_dir / "region_occlusion.csv")
    print(f"Saved explainability artifacts: {output_dir}", flush=True)
    return 0


def _temporal_occlusion_rows(model, item: dict, clip, device, class_index: int, baseline_probability: float) -> list[dict[str, object]]:
    import torch

    rows: list[dict[str, object]] = []
    time_steps = int(item["face_frames"].shape[0])
    for frame_index in range(time_steps):
        frames = item["face_frames"].clone().unsqueeze(0).to(device)
        behavior = item["behavior"].clone().unsqueeze(0).to(device)
        valid_mask = item["valid_mask"].clone().unsqueeze(0).to(device)
        frames[:, frame_index] = 0.0
        behavior[:, frame_index] = 0.0
        valid_mask[:, frame_index] = 0.0
        with torch.inference_mode():
            logits = model(frames, behavior, valid_mask)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        rows.append(
            {
                "clip_id": clip.clip_id,
                "video_id": clip.video_id,
                "frame_index": frame_index,
                "class_index": class_index,
                "baseline_probability": float(baseline_probability),
                "occluded_probability": float(probs[class_index]),
                "probability_delta": float(baseline_probability - probs[class_index]),
            }
        )
    return rows


def _region_occlusion_rows(model, item: dict, clip, config: dict, device, class_index: int, baseline_probs: np.ndarray) -> list[dict[str, object]]:
    import torch

    rows: list[dict[str, object]] = []
    for region_name in ["eyes", "mouth"]:
        frames = _occlude_region(item["face_frames"].clone(), config, region_name).unsqueeze(0).to(device)
        behavior = item["behavior"].unsqueeze(0).to(device)
        valid_mask = item["valid_mask"].unsqueeze(0).to(device)
        with torch.inference_mode():
            logits = model(frames, behavior, valid_mask)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        rows.append(
            {
                "clip_id": clip.clip_id,
                "video_id": clip.video_id,
                "region": region_name,
                "class_index": class_index,
                "baseline_probability": float(baseline_probs[class_index]),
                "occluded_probability": float(probs[class_index]),
                "probability_delta": float(baseline_probs[class_index] - probs[class_index]),
                "p_alert": float(probs[0]),
                "p_low_vigilant": float(probs[1]),
                "p_drowsy": float(probs[2]),
            }
        )
    return rows


def _occlude_region(frames, config: dict, region_name: str):
    height = int(config["image"]["height"])
    if region_name == "eyes":
        y1, y2 = int(height * 0.25), int(height * 0.50)
    elif region_name == "mouth":
        y1, y2 = int(height * 0.56), int(height * 0.84)
    else:
        raise ValueError(f"Unsupported occlusion region: {region_name}")
    frames[:, :, y1:y2, :] = 0.0
    return frames


def _write_behavior_series(clip, output_dir: Path) -> None:
    data = np.load(clip.npz_path)
    start = int(clip.start_frame)
    end = int(clip.end_frame) + 1
    rows = []
    for offset, frame_index in enumerate(range(start, end)):
        rows.append(
            {
                "clip_id": clip.clip_id,
                "video_id": clip.video_id,
                "frame_index": frame_index,
                "timestamp_seconds": float(data["timestamps"][frame_index]),
                "ear": float(data["ear"][frame_index]),
                "mar": float(data["mar"][frame_index]),
                "valid_frame": bool(data["valid_mask"][frame_index]),
                "clip_relative_frame": offset,
            }
        )
    _write_rows(rows, output_dir / f"{clip.clip_id}.csv")


def _write_gradcam_overlays(model, item: dict, clip, config: dict, device, class_index: int, output_dir: Path, frame_step: int) -> None:
    import torch

    visual_encoder = getattr(model, "visual_encoder", None)
    target_layer = getattr(visual_encoder, "layer4", None) if visual_encoder is not None else None
    if target_layer is None:
        return

    activations = {}
    gradients = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        frames = item["face_frames"].unsqueeze(0).to(device)
        behavior = item["behavior"].unsqueeze(0).to(device)
        valid_mask = item["valid_mask"].unsqueeze(0).to(device)
        # cuDNN RNNs cannot run backward from an eval-mode forward pass. Grad-CAM
        # should keep dropout/batchnorm in eval mode, so use the non-cuDNN path
        # for this tiny explanatory backward pass instead of switching the model
        # to train mode.
        cudnn_context = (
            torch.backends.cudnn.flags(enabled=False)
            if frames.is_cuda and torch.backends.cudnn.enabled
            else nullcontext()
        )
        with cudnn_context:
            logits = model(frames, behavior, valid_mask)
            logits[0, class_index].backward()
        acts = activations["value"].detach()
        grads = gradients["value"].detach()
        time_steps = int(item["face_frames"].shape[0])
        acts = acts.reshape(time_steps, acts.shape[1], acts.shape[2], acts.shape[3])
        grads = grads.reshape(time_steps, grads.shape[1], grads.shape[2], grads.shape[3])
        rgb_frames = _denormalize_frames(item["face_frames"], config)
        output_dir.mkdir(parents=True, exist_ok=True)
        for frame_index in range(0, time_steps, max(1, int(frame_step))):
            weights = grads[frame_index].mean(dim=(1, 2), keepdim=True)
            cam = torch.relu((weights * acts[frame_index]).sum(dim=0)).detach().cpu().numpy()
            if float(cam.max()) > 1e-8:
                cam = cam / float(cam.max())
            heatmap = cv2.resize(cam, (rgb_frames.shape[2], rgb_frames.shape[1]))
            heatmap_u8 = np.uint8(255 * heatmap)
            color = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
            rgb = rgb_frames[frame_index]
            overlay_bgr = cv2.addWeighted(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 0.55, color, 0.45, 0.0)
            cv2.imwrite(str(output_dir / f"{clip.clip_id}_frame_{frame_index:03d}.png"), overlay_bgr)
    finally:
        forward_handle.remove()
        backward_handle.remove()


def _denormalize_frames(frames, config: dict) -> np.ndarray:
    mean = np.asarray(config["image"].get("mean", [0.485, 0.456, 0.406]), dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(config["image"].get("std", [0.229, 0.224, 0.225]), dtype=np.float32).reshape(1, 1, 3)
    array = frames.detach().cpu().numpy().transpose(0, 2, 3, 1)
    array = (array * std + mean) * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _write_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _torch_load(path: str | Path, *, map_location):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


if __name__ == "__main__":
    raise SystemExit(main())
