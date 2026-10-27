"""Detector / online tracker adapters on fake ultralytics results (no model)."""
from types import SimpleNamespace

import numpy as np

from tracking.config import DetectorConfig
from tracking.detection import detections_from_result
from tracking.online_pass import CameraPass, PassCache
from tracking.online_tracker import YoloBotSortTracker

NAMES = {0: "ball", 1: "player", 2: "goalpost"}


class FakeBoxes:
    """Mimics ultralytics Boxes after .cpu().numpy(): arrays plus boolean-mask indexing."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = np.asarray(xyxy, float).reshape(-1, 4), np.asarray(conf, float), np.asarray(cls, float)

    def cpu(self):
        return self

    def numpy(self):
        return self

    def __getitem__(self, mask):
        return FakeBoxes(self.xyxy[mask], self.conf[mask], self.cls[mask])


def fake_result(xyxy, conf, cls):
    return SimpleNamespace(boxes=FakeBoxes(xyxy, conf, cls), names=NAMES, orig_img=np.zeros((10, 10, 3), np.uint8))


class EchoTracker:
    """Fake BoT-SORT: returns every detection it gets with track id 100 + index."""

    def __init__(self):
        self.received = None

    def update(self, boxes, image):
        self.received = boxes
        n = len(boxes.xyxy)
        return np.c_[boxes.xyxy, 100 + np.arange(n), boxes.conf, boxes.cls, np.arange(n)]


RESULT = fake_result([[0, 0, 5, 5], [10, 10, 20, 40], [30, 10, 40, 40], [1, 1, 2, 2], [50, 0, 60, 20]],
                     [0.3, 0.9, 0.8, 0.6, 0.7], [0, 1, 1, 0, 2])


def test_detections_use_model_class_names():
    detections = detections_from_result(RESULT)
    assert detections.labels.tolist() == ["ball", "player", "player", "ball", "goalpost"]
    assert len(detections.of_label("player")) == 2
    assert len(detections.of_label("referee")) == 0


def test_only_players_go_through_the_tracker_and_best_ball_is_kept():
    tracker = EchoTracker()
    adapter = YoloBotSortTracker(DetectorConfig(weights="unused.pt"))
    frame = adapter._frame_tracks(7, RESULT, tracker)
    assert len(tracker.received.xyxy) == 2
    assert frame.frame_idx == 7
    assert frame.track_ids.tolist() == [100, 101]
    assert np.allclose(frame.ball_box, [1, 1, 2, 2])           # conf 0.6 beats 0.3
    assert frame.goal_boxes.shape == (1, 4)


def test_frame_without_detections_is_empty():
    frame = YoloBotSortTracker(DetectorConfig(weights="unused.pt"))._frame_tracks(0, fake_result([], [], []),
                                                                                  EchoTracker())
    assert frame.track_ids.size == 0 and frame.ball_box is None and len(frame.goal_boxes) == 0


def test_cache_key_changes_with_detector_settings(tmp_path, settings):
    video, weights, tracker_yaml = tmp_path / "v.mp4", tmp_path / "w.pt", tmp_path / "t.yaml"
    for path in (video, weights, tracker_yaml):
        path.write_text("x")
    camera = SimpleNamespace(video=video)
    encoder = SimpleNamespace(name="enc")
    key = PassCache.key(camera, encoder, settings, DetectorConfig(weights), tracker_yaml, None)
    assert key == PassCache.key(camera, encoder, settings, DetectorConfig(weights), tracker_yaml, None)
    assert key != PassCache.key(camera, encoder, settings, DetectorConfig(weights, imgsz=1280), tracker_yaml, None)
    assert key != PassCache.key(camera, encoder, settings, DetectorConfig(weights), tracker_yaml, 50)


def test_camera_pass_is_a_plain_container():
    camera_pass = CameraPass("cam1", 3, [], {}, {})
    assert camera_pass.frame_count == 3
