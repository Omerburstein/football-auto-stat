"""Tracklets: unbroken pieces of one player's track from one camera, built from the online pass."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from tracking.appearance import AppearanceEncoder, mean_embedding
from tracking.camera import Camera
from tracking.geometry import isolated_mask
from tracking.online_tracker import FrameTracks
from tracking.settings import TrackingSettings


@dataclass
class Tracklet:
    """One continuous track. `key` is unique across cameras and splits, e.g. 'cam1:17' or 'cam1:17b'."""
    key: str
    camera_id: str
    frames: np.ndarray           # (T,) frame indices, increasing
    times: np.ndarray            # (T,) game seconds
    boxes: np.ndarray            # (T, 4) pixels
    pitch_xy: np.ndarray         # (T, 2) metres, NaN without calibration
    sample_frames: np.ndarray    # (K,) frames the embeddings were taken from
    embeddings: np.ndarray       # (K, D)

    @property
    def start(self) -> float:
        return float(self.times[0])

    @property
    def end(self) -> float:
        return float(self.times[-1])

    @property
    def duration(self) -> float:
        return self.end - self.start

    @cached_property
    def prototype(self) -> np.ndarray | None:
        """Mean appearance, or None when no clean crop was ever sampled."""
        return mean_embedding(self.embeddings) if len(self.embeddings) else None

    def box_at(self, frame_idx: int) -> np.ndarray | None:
        position = np.searchsorted(self.frames, frame_idx)
        if position < len(self.frames) and self.frames[position] == frame_idx:
            return self.boxes[position]
        return None

    def split_at(self, frame_idx: int) -> tuple[Tracklet, Tracklet]:
        """Two tracklets: frames before `frame_idx`, and from it on (keys get 'a' / 'b')."""
        cut, sample_cut = np.searchsorted(self.frames, frame_idx), np.searchsorted(self.sample_frames, frame_idx)
        halves = []
        for suffix, rows, samples in (("a", slice(0, cut), slice(0, sample_cut)),
                                      ("b", slice(cut, None), slice(sample_cut, None))):
            halves.append(Tracklet(self.key + suffix, self.camera_id, self.frames[rows], self.times[rows],
                                   self.boxes[rows], self.pitch_xy[rows], self.sample_frames[samples],
                                   self.embeddings[samples]))
        return halves[0], halves[1]


@dataclass
class _Track:
    frames: list = field(default_factory=list)
    boxes: list = field(default_factory=list)
    sample_frames: list = field(default_factory=list)
    embeddings: list = field(default_factory=list)


class TrackletBuilder:
    """Accumulates FrameTracks of one camera and samples clean appearance crops along the way."""

    def __init__(self, camera: Camera, encoder: AppearanceEncoder, settings: TrackingSettings):
        self._camera, self._encoder, self._settings = camera, encoder, settings
        self._tracks: dict[int, _Track] = {}

    def add(self, frame: FrameTracks) -> None:
        for track_id, box in zip(frame.track_ids, frame.boxes):
            track = self._tracks.setdefault(int(track_id), _Track())
            track.frames.append(frame.frame_idx)
            track.boxes.append(box)
        self._sample_embeddings(frame)

    def _sample_embeddings(self, frame: FrameTracks) -> None:
        if frame.frame_idx % self._settings.embed_every_frames or len(frame.boxes) == 0:
            return
        heights = frame.boxes[:, 3] - frame.boxes[:, 1]
        clean = isolated_mask(frame.boxes, self._settings.max_overlap_iou)
        clean &= heights >= self._settings.min_box_height_px
        if not clean.any():
            return
        embeddings = self._encoder.encode(frame.image, frame.boxes[clean])
        for track_id, embedding in zip(frame.track_ids[clean], embeddings):
            track = self._tracks[int(track_id)]
            track.sample_frames.append(frame.frame_idx)
            track.embeddings.append(embedding)

    def build(self) -> list[Tracklet]:
        """Finished tracklets, dropping ones shorter than `min_tracklet_frames`."""
        tracklets = []
        for track_id, track in sorted(self._tracks.items()):
            if len(track.frames) < self._settings.min_tracklet_frames:
                continue
            frames, boxes = np.array(track.frames), np.array(track.boxes, dtype=np.float64)
            tracklets.append(Tracklet(
                key=f"{self._camera.id}:{track_id}",
                camera_id=self._camera.id,
                frames=frames,
                times=self._camera.frame_time(frames),
                boxes=boxes,
                pitch_xy=self._camera.image_to_pitch(Camera.foot_points(boxes)),
                sample_frames=np.array(track.sample_frames, dtype=int),
                embeddings=np.array(track.embeddings) if track.embeddings else np.empty((0, 0)),
            ))
        return tracklets
