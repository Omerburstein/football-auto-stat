"""Single-frame object detection (used by the enrollment tool and wrapped by the online tracker)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from tracking.config import DetectorConfig


@dataclass
class Detections:
    """Detections of one image (empty arrays when nothing was found)."""
    boxes: np.ndarray        # (N, 4) pixels
    scores: np.ndarray       # (N,)
    labels: np.ndarray       # (N,) class names from the model, e.g. "player"

    def of_label(self, label: str) -> Detections:
        mask = self.labels == label
        return Detections(self.boxes[mask], self.scores[mask], self.labels[mask])

    def __len__(self) -> int:
        return len(self.boxes)


class Detector(ABC):
    """Finds objects in one image. Swap implementations to try other models."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> Detections:
        ...


class YoloDetector(Detector):
    """Ultralytics YOLO model; class names come from `model.names`, never hardcoded ids."""

    def __init__(self, config: DetectorConfig):
        from ultralytics import YOLO  # heavy import, only when a model is really used
        self._model = YOLO(str(config.weights))
        self._config = config

    def detect(self, image: np.ndarray) -> Detections:
        result = self._model.predict(image, conf=self._config.conf, imgsz=self._config.imgsz, verbose=False)[0]
        return detections_from_result(result)


def detections_from_result(result) -> Detections:
    """Convert an ultralytics `Results` object into Detections."""
    boxes = result.boxes.cpu().numpy()
    labels = np.array([result.names[int(c)] for c in boxes.cls], dtype=object)
    return Detections(boxes.xyxy.astype(np.float64), boxes.conf.astype(np.float64), labels)
