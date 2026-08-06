from __future__ import annotations

from drowsiness_detection.capture.camera import _is_stream_url, _parse_source


def test_numeric_source_string_becomes_camera_index() -> None:
    assert _parse_source("0") == 0


def test_ip_camera_url_is_stream_source() -> None:
    assert _is_stream_url("http://192.168.0.101:8080/video")
    assert _is_stream_url("rtsp://192.168.0.101/live")
    assert not _is_stream_url("data/example.mp4")

