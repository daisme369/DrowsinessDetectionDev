from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drowsiness_detection.video import LABEL_TO_ID, load_video_config, save_resolved_config
from drowsiness_detection.video.clips import read_clip_manifest
from drowsiness_detection.video.dataset import TemporalClipDataset
from drowsiness_detection.video.evaluation import (
    class_weights_from_clips,
    default_behavior_stats_path,
    default_clips_path,
    evaluate_model_on_loader,
    load_behavior_stats,
    make_dataloader,
)
from drowsiness_detection.video.model import build_baseline_d_model, freeze_visual_encoder, unfreeze_visual_stage
from drowsiness_detection.video.runtime import collect_environment, git_commit, set_reproducible_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Baseline D temporal video model.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "experiment_e4.yaml"))
    parser.add_argument("--clips", default=None)
    parser.add_argument("--behavior-stats", default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max-train-clips", type=int, default=0)
    parser.add_argument("--max-validation-clips", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_video_config(args.config)
    print(json.dumps(config, indent=2), flush=True)
    set_reproducible_seed(int(config["training"].get("random_seed", config["project"].get("random_seed", 42))))

    import torch
    from torch import nn

    clips_path = Path(args.clips) if args.clips else default_clips_path(config)
    if not clips_path.exists():
        raise FileNotFoundError(f"Clip manifest does not exist. Run scripts/preprocess.py first: {clips_path}")
    behavior_stats_path = Path(args.behavior_stats) if args.behavior_stats else default_behavior_stats_path(config)
    behavior_stats = load_behavior_stats(behavior_stats_path)
    clips = read_clip_manifest(clips_path)
    train_clips = [clip for clip in clips if clip.split == "train"]
    validation_clips = [clip for clip in clips if clip.split == "validation"]
    if args.max_train_clips > 0:
        train_clips = train_clips[: args.max_train_clips]
    if args.max_validation_clips > 0:
        validation_clips = validation_clips[: args.max_validation_clips]
    if not train_clips:
        raise ValueError(f"No training clips found in {clips_path}")
    if not validation_clips:
        raise ValueError(f"No validation clips found in {clips_path}")

    train_dataset = TemporalClipDataset(train_clips, config, behavior_stats=behavior_stats, training=True)
    validation_dataset = TemporalClipDataset(validation_clips, config, behavior_stats=behavior_stats, training=False)
    train_loader = make_dataloader(train_dataset, config, shuffle=True)
    validation_loader = make_dataloader(validation_dataset, config, shuffle=False)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_baseline_d_model(config).to(device)
    freeze_epochs = int(config["model"]["cnn"].get("initial_freeze_epochs", 0))
    start_epoch = 1
    best_metric = -1.0
    checkpoint = None
    if args.resume:
        checkpoint = _torch_load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_metric = float(checkpoint.get("best_metric", -1.0))
    _apply_visual_trainability(model, config, epoch=start_epoch, freeze_epochs=freeze_epochs)

    class_weights = class_weights_from_clips(train_clips, config)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = _build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(1, int(config["training"].get("early_stopping_patience", 7)) // 2),
    )
    use_amp = bool(config["training"].get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    run_dir = _create_run_dir(config)
    save_resolved_config(config, run_dir / "config_resolved.yaml")
    (run_dir / "environment.json").write_text(json.dumps(collect_environment(), indent=2), encoding="utf-8")
    _copy_if_exists(args.split_manifest or _default_split_manifest_path(config), run_dir / "split_manifest.json")

    checkpoint_dir = Path(config["training"].get("checkpoint_dir", ROOT / "checkpoints" / "baseline_d"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_video_macro_f1.pt"
    last_path = checkpoint_dir / "last.pt"

    max_epochs = int(args.epochs or config["training"]["maximum_epochs"])
    patience = int(config["training"].get("early_stopping_patience", 7))
    stale_epochs = 0
    log_rows: list[dict[str, object]] = []
    print(f"Training clips={len(train_dataset)} validation clips={len(validation_dataset)} device={device}", flush=True)

    for epoch in range(start_epoch, max_epochs + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            _apply_visual_trainability(model, config, epoch=epoch, freeze_epochs=freeze_epochs)
            optimizer = _build_optimizer(model, config)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=max(1, patience // 2))

        train_loss = _train_one_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer,
            scaler,
            use_amp=use_amp,
            gradient_clip_norm=float(config["training"].get("gradient_clip_norm", 0.0)),
        )
        validation_result = evaluate_model_on_loader(model, validation_loader, device, criterion)
        video_macro_f1 = float(validation_result["video_metrics"].macro_f1)
        scheduler.step(video_macro_f1)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_result["loss"],
            "validation_clip_macro_f1": validation_result["clip_metrics"].macro_f1,
            "validation_video_macro_f1": video_macro_f1,
            "validation_video_accuracy": validation_result["video_metrics"].accuracy,
        }
        log_rows.append(row)
        _write_training_log(log_rows, run_dir / "training_log.csv")
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"val_video_macro_f1={video_macro_f1:.4f}",
            flush=True,
        )

        checkpoint_payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            config=config,
            behavior_stats=behavior_stats,
            split_manifest_path=args.split_manifest or str(_default_split_manifest_path(config)),
            best_metric=max(best_metric, video_macro_f1),
        )
        torch.save(checkpoint_payload, last_path)
        if video_macro_f1 > best_metric:
            best_metric = video_macro_f1
            stale_epochs = 0
            torch.save(checkpoint_payload, best_path)
            (run_dir / "best_metrics.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "clip": validation_result["clip_metrics"].to_dict(),
                        "video": validation_result["video_metrics"].to_dict(),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (run_dir / "checkpoint_reference.txt").write_text(str(best_path), encoding="utf-8")
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {patience} stale validation epochs.", flush=True)
                break

    print(f"Saved best checkpoint: {best_path}", flush=True)
    print(f"Run artifacts: {run_dir}", flush=True)
    return 0


def _train_one_epoch(model, dataloader, device, criterion, optimizer, scaler, *, use_amp: bool, gradient_clip_norm: float) -> float:
    import torch

    model.train()
    losses: list[float] = []
    for batch in dataloader:
        frames = batch["face_frames"].to(device, non_blocking=True)
        behavior = batch["behavior"].to(device, non_blocking=True)
        valid_mask = batch["valid_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(frames, behavior, valid_mask)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        if gradient_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(sum(losses) / max(len(losses), 1))


def _build_optimizer(model, config: dict):
    import torch

    cnn_lr = float(config["training"]["cnn_learning_rate"])
    temporal_lr = float(config["training"]["temporal_learning_rate"])
    classifier_lr = float(config["training"]["classifier_learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])
    groups = [
        {
            "params": [parameter for name, parameter in model.named_parameters() if name.startswith("visual_encoder") and parameter.requires_grad],
            "lr": cnn_lr,
        },
        {
            "params": [parameter for name, parameter in model.named_parameters() if name.startswith("lstm") and parameter.requires_grad],
            "lr": temporal_lr,
        },
        {
            "params": [parameter for name, parameter in model.named_parameters() if name.startswith("classifier") and parameter.requires_grad],
            "lr": classifier_lr,
        },
    ]
    groups = [group for group in groups if group["params"]]
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def _apply_visual_trainability(model, config: dict, *, epoch: int, freeze_epochs: int) -> None:
    if freeze_epochs > 0 and epoch <= freeze_epochs:
        freeze_visual_encoder(model, True)
        return
    stages = list(config["model"]["cnn"].get("unfreeze_stages", []))
    if not stages:
        freeze_visual_encoder(model, False)
        return
    freeze_visual_encoder(model, True)
    for stage in stages:
        unfreeze_visual_stage(model, str(stage))


def _checkpoint_payload(*, model, optimizer, scheduler, epoch: int, config: dict, behavior_stats: dict, split_manifest_path: str, best_metric: float) -> dict:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "configuration": config,
        "label_mapping": LABEL_TO_ID,
        "behavior_normalization_statistics": behavior_stats,
        "split_manifest_path": split_manifest_path,
        "git_commit": git_commit(),
        "best_metric": float(best_metric),
    }


def _create_run_dir(config: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{config.get('experiment', {}).get('name', 'baseline_d')}"
    run_dir = Path(config["project"]["output_dir"]) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _write_training_log(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _default_split_manifest_path(config: dict) -> Path:
    split = config["split"]
    output_dir = Path(split.get("output_dir", "artifacts/splits"))
    seed = int(split.get("random_seed", config["project"].get("random_seed", 42)))
    if split.get("mode", "fixed") == "fixed":
        return output_dir / f"fixed_split_seed_{seed}.json"
    return output_dir / "fold_1.json"


def _copy_if_exists(source: str | Path, destination: Path) -> None:
    source_path = Path(source)
    if source_path.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)


def _torch_load(path: str | Path, *, map_location):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


if __name__ == "__main__":
    raise SystemExit(main())
