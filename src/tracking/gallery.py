"""Per-player appearance gallery, kept separately for every outfit a player wears during the game."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from tracking.appearance import AppearanceEncoder, cosine_distance
from tracking.camera import Camera
from tracking.enrollment import Enrollment
from tracking.timeline import Timeline

GalleryKey = tuple[str, int]   # (player name, outfit period)


class PlayerGallery:
    """Embeddings of known views of each (player, outfit period)."""

    def __init__(self):
        self._embeddings: dict[GalleryKey, list[np.ndarray]] = defaultdict(list)

    @classmethod
    def from_enrollment(cls, enrollment: Enrollment, cameras: dict[str, Camera],
                        encoder: AppearanceEncoder, timeline: Timeline) -> PlayerGallery:
        """Encode every enrolled box. Each frame is read once even when it has several samples."""
        gallery = cls()
        by_frame = defaultdict(list)
        for sample in enrollment.samples:
            by_frame[(sample.camera_id, sample.frame)].append(sample)
        for (camera_id, frame), samples in sorted(by_frame.items()):
            camera = cameras[camera_id]
            image = camera.read_frame(frame)
            if image is None:
                raise ValueError(f"enrollment frame {frame} is past the end of camera {camera_id}")
            embeddings = encoder.encode(image, np.array([s.box for s in samples]))
            time = float(camera.frame_time(frame))
            for sample, embedding in zip(samples, embeddings):
                gallery.add(sample.name, timeline.outfit_period(sample.name, time), embedding)
        return gallery

    def add(self, name: str, period: int, embedding: np.ndarray) -> None:
        self._embeddings[(name, period)].append(np.asarray(embedding))

    def copy(self) -> PlayerGallery:
        clone = PlayerGallery()
        for key, embeddings in self._embeddings.items():
            clone._embeddings[key] = list(embeddings)
        return clone

    def has(self, name: str, period: int) -> bool:
        return bool(self._embeddings.get((name, period)))

    def keys(self) -> list[GalleryKey]:
        return [key for key, embeddings in self._embeddings.items() if embeddings]

    def distance(self, name: str, period: int, prototype: np.ndarray) -> float:
        """Distance from an appearance to the closest known view of the player in that outfit;
        inf when that outfit was never enrolled (the player can't be recognised then)."""
        embeddings = self._embeddings.get((name, period))
        if not embeddings:
            return np.inf
        return float(cosine_distance(prototype, np.array(embeddings)).min())
