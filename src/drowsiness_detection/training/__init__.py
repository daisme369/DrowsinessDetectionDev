"""Training helpers for lightweight drowsiness baselines."""

from .datasets import RoiSample, YoloRoiDataset, collect_yolo_roi_samples, count_labels
from .feature_vectors import (
    DEFAULT_GEOMETRIC_FEATURE_NAMES,
    build_frame_feature_dict,
    feature_matrix_from_rows,
    feature_vector_from_dict,
)
from .metrics import ClassificationMetrics, compute_classification_metrics
try:
    from .video_sequences import (
        VideoSample,
        build_sequence_windows,
        discover_video_samples,
        parse_video_level,
        sequence_vector_from_feature_dict,
    )
except ModuleNotFoundError:
    VideoSample = None
    build_sequence_windows = None
    discover_video_samples = None
    parse_video_level = None
    sequence_vector_from_feature_dict = None

__all__ = [
    "ClassificationMetrics",
    "DEFAULT_GEOMETRIC_FEATURE_NAMES",
    "RoiSample",
    "YoloRoiDataset",
    "build_frame_feature_dict",
    "collect_yolo_roi_samples",
    "compute_classification_metrics",
    "count_labels",
    "feature_matrix_from_rows",
    "feature_vector_from_dict",
]

if VideoSample is not None:
    __all__.extend(
        [
            "VideoSample",
            "build_sequence_windows",
            "discover_video_samples",
            "parse_video_level",
            "sequence_vector_from_feature_dict",
        ]
    )
