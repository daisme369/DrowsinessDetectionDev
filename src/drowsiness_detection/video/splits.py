from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable

from drowsiness_detection.video.manifest import VideoRecord


@dataclass(frozen=True, slots=True)
class SplitManifest:
    train_subjects: list[str]
    validation_subjects: list[str]
    test_subjects: list[str]
    train_video_ids: list[str]
    validation_video_ids: list[str]
    test_video_ids: list[str]


def create_fixed_subject_split(
    records: Iterable[VideoRecord],
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> SplitManifest:
    subjects = sorted({record.subject_id for record in records})
    rng = random.Random(seed)
    rng.shuffle(subjects)
    subject_count = len(subjects)
    train_count = int(round(subject_count * train_ratio))
    validation_count = int(round(subject_count * validation_ratio))
    test_count = subject_count - train_count - validation_count
    if min(train_count, validation_count, test_count) <= 0:
        raise ValueError("Split ratios must assign at least one subject to each split")
    train_subjects = sorted(subjects[:train_count])
    validation_subjects = sorted(subjects[train_count : train_count + validation_count])
    test_subjects = sorted(subjects[train_count + validation_count :])
    return _build_manifest(records, train_subjects, validation_subjects, test_subjects)


def create_five_fold_subject_splits(
    records: Iterable[VideoRecord],
    *,
    number_of_folds: int,
    seed: int,
    validation_subject_count: int = 6,
) -> list[SplitManifest]:
    records = list(records)
    subjects = sorted({record.subject_id for record in records})
    rng = random.Random(seed)
    shuffled = list(subjects)
    rng.shuffle(shuffled)
    folds = _round_robin_folds(shuffled, number_of_folds)
    manifests: list[SplitManifest] = []
    for fold_index, test_subjects in enumerate(folds):
        remaining = sorted(set(subjects).difference(test_subjects))
        validation_subjects = sorted(remaining[:validation_subject_count])
        train_subjects = sorted(set(remaining).difference(validation_subjects))
        manifests.append(_build_manifest(records, train_subjects, validation_subjects, sorted(test_subjects)))
    return manifests


def assert_no_split_leakage(manifest: SplitManifest) -> None:
    _assert_disjoint("subjects", manifest.train_subjects, manifest.validation_subjects, manifest.test_subjects)
    _assert_disjoint("videos", manifest.train_video_ids, manifest.validation_video_ids, manifest.test_video_ids)


def write_split_manifest(manifest: SplitManifest, path: str | Path) -> None:
    assert_no_split_leakage(manifest)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")


def read_split_manifest(path: str | Path) -> SplitManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return SplitManifest(
        train_subjects=list(payload["train_subjects"]),
        validation_subjects=list(payload["validation_subjects"]),
        test_subjects=list(payload["test_subjects"]),
        train_video_ids=list(payload["train_video_ids"]),
        validation_video_ids=list(payload["validation_video_ids"]),
        test_video_ids=list(payload["test_video_ids"]),
    )


def split_for_video_id(manifest: SplitManifest, video_id: str) -> str:
    if video_id in set(manifest.train_video_ids):
        return "train"
    if video_id in set(manifest.validation_video_ids):
        return "validation"
    if video_id in set(manifest.test_video_ids):
        return "test"
    raise KeyError(f"Video id is not present in split manifest: {video_id}")


def _build_manifest(
    records: Iterable[VideoRecord],
    train_subjects: list[str],
    validation_subjects: list[str],
    test_subjects: list[str],
) -> SplitManifest:
    records = list(records)
    train_set = set(train_subjects)
    validation_set = set(validation_subjects)
    test_set = set(test_subjects)
    manifest = SplitManifest(
        train_subjects=train_subjects,
        validation_subjects=validation_subjects,
        test_subjects=test_subjects,
        train_video_ids=sorted(record.video_id for record in records if record.subject_id in train_set),
        validation_video_ids=sorted(record.video_id for record in records if record.subject_id in validation_set),
        test_video_ids=sorted(record.video_id for record in records if record.subject_id in test_set),
    )
    assert_no_split_leakage(manifest)
    return manifest


def _round_robin_folds(subjects: list[str], number_of_folds: int) -> list[list[str]]:
    folds = [[] for _ in range(number_of_folds)]
    for index, subject in enumerate(subjects):
        folds[index % number_of_folds].append(subject)
    return [sorted(fold) for fold in folds]


def _assert_disjoint(name: str, train: list[str], validation: list[str], test: list[str]) -> None:
    train_set = set(train)
    validation_set = set(validation)
    test_set = set(test)
    if not train_set.isdisjoint(validation_set):
        raise AssertionError(f"Train/validation {name} overlap: {sorted(train_set & validation_set)}")
    if not train_set.isdisjoint(test_set):
        raise AssertionError(f"Train/test {name} overlap: {sorted(train_set & test_set)}")
    if not validation_set.isdisjoint(test_set):
        raise AssertionError(f"Validation/test {name} overlap: {sorted(validation_set & test_set)}")
