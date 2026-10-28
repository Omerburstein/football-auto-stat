"""A fixed camera: its video, timing and (optional) calibration to pitch metres."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tracking.calibration import Calibration
from tracking.config import CameraConfig


class Camera:
    """Reads frames of one camera's video and maps them to game time and pitch positions."""

    def __init__(self, config: CameraConfig, calibration: Calibration | None = None, fps: float | None = None):
        self.config = config
        self.calibration = calibration
        self._fps = fps

    @classmethod
    def from_config(cls, config: CameraConfig) -> Camera:
        """Camera with its calibration file loaded when it exists (positions are NaN otherwise)."""
        calibration = Calibration.load(config.calibration) if Path(config.calibration).exists() else None
        return cls(config, calibration)

    @property
    def id(self) -> str:
        return self.config.id

    @property
    def video(self) -> Path:
        return self.config.video

    @property
    def fps(self) -> float:
        if self._fps is None:
            capture = self._open()
            self._fps = capture.get(cv2.CAP_PROP_FPS) or 25.0  # some phone files report 0
            capture.release()
        return self._fps

    def frame_time(self, frame_idx) -> float | np.ndarray:
        """Game time (s) of a frame index, including this camera's sync offset."""
        return np.asarray(frame_idx) / self.fps + self.config.time_offset_s

    def time_to_frame(self, t: float) -> int:
        return max(round((t - self.config.time_offset_s) * self.fps), 0)

    def read_frame(self, frame_idx: int) -> np.ndarray | None:
        """BGR image of one frame, or None past the end of the video."""
        capture = self._open()
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, image = capture.read()
        capture.release()
        return image if ok else None

    def image_to_pitch(self, points) -> np.ndarray:
        """(N, 2) pixels -> (N, 2) metres; all NaN when the camera is not calibrated."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if self.calibration is None:
            return np.full_like(points, np.nan)
        return self.calibration.image_to_pitch(points)

    @staticmethod
    def foot_points(boxes) -> np.ndarray:
        """Bottom-centre of each (x1, y1, x2, y2) box - the point that touches the pitch."""
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(str(self.video))
        if not capture.isOpened():
            raise FileNotFoundError(f"cannot open video {self.video}")
        return capture
