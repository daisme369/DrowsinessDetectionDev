from __future__ import annotations

from drowsiness_detection.video.aggregation import ClipPrediction, aggregate_video_predictions


def test_video_aggregation_uses_mean_probability_per_video() -> None:
    clips = [
        ClipPrediction("c1", "v1", "s1", 2, 0.1, 0.2, 0.7, 2, 0.0, 8.0),
        ClipPrediction("c2", "v1", "s1", 2, 0.2, 0.1, 0.7, 2, 5.6, 13.6),
        ClipPrediction("c3", "v2", "s2", 0, 0.8, 0.1, 0.1, 0, 0.0, 8.0),
    ]

    videos = aggregate_video_predictions(clips)

    assert len(videos) == 2
    assert videos[0].video_id == "v1"
    assert videos[0].predicted_label == 2
    assert videos[0].num_clips == 2
    assert round(videos[0].p_drowsy, 6) == 0.7
    assert round(videos[0].p_alert + videos[0].p_low_vigilant + videos[0].p_drowsy, 6) == 1.0
    assert videos[1].video_id == "v2"
    assert videos[1].predicted_label == 0
