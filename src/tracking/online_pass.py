"""The slow, model-based stage: detect + track + sample appearance over a whole video, cached on disk.

Everything after it (identity assignment, game state, report) takes seconds, so editing match.yaml
or the enrollment and re-running reuses this cache.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tracking.appearance import AppearanceEncoder
from tracking.camera import Camera
from tracking.config import DetectorConfig
from tracking.online_tracker import OnlineTracker
from tracking.settings import TrackingSettings
from tracking.tracklet import Tracklet, TrackletBuilder

PROGRESS_EVERY_FRAMES = 250


@dataclass
class CameraPass:
    """Result of the online pass over one camera's video."""
    camera_id: str
    frame_count: int
    tracklets: list[Tracklet]
    ball_boxes: dict[int, np.ndarray]          # frame -> most confident ball box
    goal_boxes: dict[int, np.ndarray]          # frame -> (M, 4) goal boxes


def run_online_pass(camera: Camera, tracker: OnlineTracker, encoder: AppearanceEncoder,
                    settings: TrackingSettings, max_frames: int | None = None) -> CameraPass:
    """Track one camera's whole video and collect tracklets, ball and goal boxes."""
    builder = TrackletBuilder(camera, encoder, settings)
    ball, goals, frame_count = {}, {}, 0
    for frame in tracker.run(camera, max_frames):
        builder.add(frame)
        frame_count = frame.frame_idx + 1
        if frame.ball_box is not None:
            ball[frame.frame_idx] = frame.ball_box
        if len(frame.goal_boxes):
            goals[frame.frame_idx] = frame.goal_boxes
        if frame.frame_idx % PROGRESS_EVERY_FRAMES == 0:
            print(f"  {camera.id}: frame {frame.frame_idx}", flush=True)
    return CameraPass(camera.id, frame_count, builder.build(), ball, goals)


class PassCache:
    """Pickled CameraPass per camera in the run folder, reused while its inputs are unchanged."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)

    def get_or_run(self, camera: Camera, tracker: OnlineTracker, encoder: AppearanceEncoder,
                   settings: TrackingSettings, detector: DetectorConfig, tracker_config: Path,
                   max_frames: int | None = None, refresh: bool = False) -> CameraPass:
        """The cached CameraPass when its inputs are unchanged, else run the online pass and cache it."""
        key = self.key(camera, encoder, settings, detector, tracker_config, max_frames)
        path = self.run_dir / f"online_{camera.id}.pkl"
        if path.exists() and not refresh:
            cached = pickle.loads(path.read_bytes())
            if cached.get("key") == key:
                print(f"  {camera.id}: using cached online pass ({path.name})")
                return cached["pass"]
        result = run_online_pass(camera, tracker, encoder, settings, max_frames)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({"key": key, "pass": result}))
        return result

    @staticmethod
    def key(camera: Camera, encoder: AppearanceEncoder, settings: TrackingSettings,
            detector: DetectorConfig, tracker_config: Path, max_frames: int | None) -> str:
        """Hash of every input that changes the online pass. Calibration and time offset are not
        part of it: pitch positions and times are recomputed after loading (refresh_geometry)."""
        video, weights = Path(camera.video), Path(detector.weights)
        inputs = {
            "video": [str(video.resolve()), video.stat().st_size, video.stat().st_mtime],
            "weights": [str(weights.resolve()), weights.stat().st_mtime],
            "detector": [detector.conf, detector.imgsz, detector.player_label, detector.ball_label,
                         detector.goal_label],
            "tracker": Path(tracker_config).read_text(encoding="utf-8"),
            "encoder": encoder.name,
            "sampling": [settings.embed_every_frames, settings.min_box_height_px,
                         settings.max_overlap_iou, settings.min_tracklet_frames],
            "max_frames": max_frames,
        }
        return hashlib.sha1(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def refresh_geometry(tracklets: list[Tracklet], camera: Camera) -> None:
    """Recompute times and pitch positions with the camera's current offset and calibration."""
    for tracklet in tracklets:
        tracklet.times = np.asarray(camera.frame_time(tracklet.frames), dtype=np.float64)
        tracklet.pitch_xy = camera.image_to_pitch(Camera.foot_points(tracklet.boxes))
