from __future__ import annotations

from drowsiness_detection.video.manifest import VideoRecord
from drowsiness_detection.video.splits import create_five_fold_subject_splits, create_fixed_subject_split


def test_fixed_split_keeps_subjects_disjoint_and_uses_42_9_9_subjects() -> None:
    records = _records()

    split = create_fixed_subject_split(records, train_ratio=0.70, validation_ratio=0.15, test_ratio=0.15, seed=42)

    assert len(split.train_subjects) == 42
    assert len(split.validation_subjects) == 9
    assert len(split.test_subjects) == 9
    assert set(split.train_subjects).isdisjoint(split.validation_subjects)
    assert set(split.train_subjects).isdisjoint(split.test_subjects)
    assert set(split.validation_subjects).isdisjoint(split.test_subjects)
    assert all(video_id.startswith(f"S{subject}_") for subject in split.train_subjects for video_id in split.train_video_ids if video_id.startswith(f"S{subject}_"))


def test_five_fold_uses_each_subject_as_test_once() -> None:
    records = _records()

    folds = create_five_fold_subject_splits(records, number_of_folds=5, seed=42)

    assert len(folds) == 5
    all_test_subjects = []
    for fold in folds:
        assert len(fold.test_subjects) == 12
        assert len(fold.validation_subjects) == 6
        assert len(fold.train_subjects) == 42
        assert set(fold.train_subjects).isdisjoint(fold.validation_subjects)
        assert set(fold.train_subjects).isdisjoint(fold.test_subjects)
        assert set(fold.validation_subjects).isdisjoint(fold.test_subjects)
        all_test_subjects.extend(fold.test_subjects)
    assert sorted(all_test_subjects) == [f"{subject:02d}" for subject in range(1, 61)]


def _records() -> list[VideoRecord]:
    records: list[VideoRecord] = []
    labels = [(0, "alert", 0), (5, "low_vigilant", 1), (10, "drowsy", 2)]
    for subject in range(1, 61):
        subject_id = f"{subject:02d}"
        for raw_label, label_name, label_id in labels:
            records.append(
                VideoRecord(
                    video_id=f"S{subject_id}_{label_name}",
                    video_path=f"data/video/{subject_id}/{raw_label}.mp4",
                    subject_id=subject_id,
                    label_name=label_name,
                    label_id=label_id,
                    raw_label=raw_label,
                    fold_id="",
                    duration_seconds=10.0,
                    original_fps=30.0,
                    frame_count=300,
                    width=640,
                    height=480,
                )
            )
    return records
