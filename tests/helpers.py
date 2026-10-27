"""Shared fixtures: synthetic tracklets, outfits and a tiny synthetic match (video + fake tracker)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from tracking.online_tracker import FrameTracks, OnlineTracker
from tracking.settings import TrackingSettings
from tracking.tracklet import Tracklet

FPS = 10.0
GRASS_BGR = (40, 140, 40)


def unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


# Orthogonal "outfits" for identity tests that bypass the image encoder.
OUTFITS = {name: unit(np.eye(6)[i]) for i, name in enumerate(["red", "blue", "white", "black", "yellow", "orange"])}


def make_tracklet(key: str, start_frame: int, n_frames: int, embedding=None, camera: str = "cam1",
                  x_start: float = 0.0, x_step: float = 0.0, y: float = 10.0, every: int = 5,
                  fps: float = FPS, noise: float = 0.0, seed: int = 0) -> Tracklet:
    """Tracklet of a player walking along x (metres, x_step per frame) with one embedding every `every` frames."""
    frames = np.arange(start_frame, start_frame + n_frames)
    xs = x_start + x_step * np.arange(n_frames)
    pitch_xy = np.stack([xs, np.full(n_frames, y)], axis=1)
    boxes = np.stack([xs * 10, np.full(n_frames, y * 10), xs * 10 + 20, np.full(n_frames, y * 10 + 50)], axis=1)
    sample_frames = frames[::every]
    if embedding is None:
        embeddings = np.empty((0, 0))
    else:
        rng = np.random.default_rng(seed)
        embeddings = np.array([unit(embedding + noise * rng.standard_normal(len(embedding))) for _ in sample_frames])
    return Tracklet(key, camera, frames, frames / fps, boxes, pitch_xy, sample_frames, embeddings)


@pytest.fixture
def settings() -> TrackingSettings:
    return TrackingSettings()


# ---------------------------------------------------------------- synthetic match for integration tests

# name -> (shirt BGR, shorts BGR, x start px, x speed px/frame)
SCENE_PLAYERS = {
    "Red": ((30, 30, 220), (255, 255, 255), 100, 3),
    "Blue": ((220, 60, 30), (20, 20, 20), 500, -2),
    "Yellow": ((30, 220, 230), (220, 60, 30), 300, 1),
}
SCENE_FRAMES = 60
SCENE_SIZE = (360, 640)   # height, width
PLAYER_SIZE = (60, 24)    # height, width px
BREAK_FRAME = 30          # the online tracker loses every player here and starts new ids


def scene_boxes(frame: int) -> dict[str, np.ndarray]:
    boxes = {}
    for i, (name, (_, _, x0, speed)) in enumerate(SCENE_PLAYERS.items()):
        x, y = x0 + speed * frame, 80 + 90 * i
        boxes[name] = np.array([x, y, x + PLAYER_SIZE[1], y + PLAYER_SIZE[0]], dtype=np.float64)
    return boxes


def scene_image(frame: int) -> np.ndarray:
    image = np.full((*SCENE_SIZE, 3), GRASS_BGR, dtype=np.uint8)
    for name, box in scene_boxes(frame).items():
        shirt, shorts = SCENE_PLAYERS[name][:2]
        x1, y1, x2, y2 = box.astype(int)
        middle = (y1 + y2) // 2
        image[y1:middle, x1:x2] = shirt
        image[middle:y2, x1:x2] = shorts
    return image


class SceneTracker(OnlineTracker):
    """Fake online tracker: perfect boxes, but every track id changes at BREAK_FRAME."""

    def run(self, camera, max_frames=None):
        for frame in range(SCENE_FRAMES if max_frames is None else min(max_frames, SCENE_FRAMES)):
            boxes = scene_boxes(frame)
            offset = 10 if frame >= BREAK_FRAME else 0
            yield FrameTracks(frame, scene_image(frame), np.arange(len(boxes)) + 1 + offset,
                              np.array(list(boxes.values())), None, np.empty((0, 4)))


@pytest.fixture
def scene_match(tmp_path) -> Path:
    """match.yaml + lossless video of the synthetic scene + enrollment of every player at frame 0."""
    video = tmp_path / "scene.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"FFV1"), FPS, SCENE_SIZE[::-1])
    for frame in range(SCENE_FRAMES):
        writer.write(scene_image(frame))
    writer.release()
    samples = [{"name": name, "camera": "cam1", "frame": 0, "box": box.tolist()}
               for name, box in scene_boxes(0).items()]
    (tmp_path / "enrollment.yaml").write_text(yaml.safe_dump({"samples": samples}), encoding="utf-8")
    match = {
        "max_players": 3, "pitch": {"length_m": 40, "width_m": 20},
        "cameras": [{"id": "cam1", "video": "scene.avi"}],
        "roster": list(SCENE_PLAYERS),
        "timeline": [{"t": 0, "teams": {"A": ["Red", "Yellow"], "B": ["Blue"]}, "goalkeepers": {"B": "Blue"}}],
    }
    path = tmp_path / "match.yaml"
    path.write_text(yaml.safe_dump(match), encoding="utf-8")
    return path
