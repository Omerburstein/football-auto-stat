"""Frame-by-frame tracking: links player detections of consecutive frames into short tracks.

The online tracker is tuned conservatively - it should rather break a track than swap two players,
because the offline identity stage (tracking/identity.py) re-joins broken tracks but cannot easily
undo a swap.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tracking.camera import Camera
from tracking.config import DetectorConfig
from tracking.detection import detections_from_result

DEFAULT_TRACKER_CONFIG = Path(__file__).resolve().parent / "trackers" / "botsort_fixed_camera.yaml"


@dataclass
class FrameTracks:
    """Everything the online pass found in one frame."""
    frame_idx: int
    image: np.ndarray
    track_ids: np.ndarray            # (N,) online track id per player box
    boxes: np.ndarray                # (N, 4) player boxes, pixels
    ball_box: np.ndarray | None      # most confident ball detection, or None
    goal_boxes: np.ndarray           # (M, 4)


class OnlineTracker(ABC):
    """Yields the tracked players (plus ball and goals) for every frame of a camera."""

    @abstractmethod
    def run(self, camera: Camera, max_frames: int | None = None) -> Iterator[FrameTracks]:
        ...


class YoloBotSortTracker(OnlineTracker):
    """YOLO detections + ultralytics BoT-SORT on the player class only.

    Only players go through the tracker so ball and goal detections are never dropped by it.
    """

    def __init__(self, config: DetectorConfig, tracker_config: Path = DEFAULT_TRACKER_CONFIG):
        self._config = config
        self._tracker_config = Path(tracker_config)

    def run(self, camera: Camera, max_frames: int | None = None) -> Iterator[FrameTracks]:
        """Frame-by-frame player tracks; a fresh tracker per call so cameras never share ids."""
        from ultralytics import YOLO
        model = YOLO(str(self._config.weights))
        tracker = self._new_tracker(camera.fps)
        results = model.predict(str(camera.video), stream=True, conf=self._config.conf,
                                imgsz=self._config.imgsz, verbose=False)
        for frame_idx, result in enumerate(results):
            if max_frames is not None and frame_idx >= max_frames:
                break
            yield self._frame_tracks(frame_idx, result, tracker)

    def _new_tracker(self, fps: float):
        from ultralytics.trackers.bot_sort import BOTSORT
        from ultralytics.utils import YAML, IterableSimpleNamespace
        args = IterableSimpleNamespace(**YAML.load(self._tracker_config))
        return BOTSORT(args=args, frame_rate=round(fps))

    def _frame_tracks(self, frame_idx: int, result, tracker) -> FrameTracks:
        detections = detections_from_result(result)
        boxes = result.boxes.cpu().numpy()
        tracks = tracker.update(boxes[detections.labels == self._config.player_label], result.orig_img)
        tracks = np.asarray(tracks).reshape(-1, 8)  # x1, y1, x2, y2, track_id, score, cls, det_idx
        ball = detections.of_label(self._config.ball_label)
        return FrameTracks(
            frame_idx=frame_idx,
            image=result.orig_img,
            track_ids=tracks[:, 4].astype(int),
            boxes=tracks[:, :4].astype(np.float64),
            ball_box=ball.boxes[np.argmax(ball.scores)] if len(ball) else None,
            goal_boxes=detections.of_label(self._config.goal_label).boxes,
        )
