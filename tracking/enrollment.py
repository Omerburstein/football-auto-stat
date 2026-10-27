"""Enrollment: the user's clicks saying "this box, at this frame, is <name>" (tools/enroll.py).

Each sample teaches the gallery one view of a player's outfit and pins the tracklet that
contains the box to that name.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from tracking.config import format_time


@dataclass
class EnrollmentSample:
    name: str
    camera_id: str
    frame: int
    box: np.ndarray    # (4,) pixels


class Enrollment:
    """All samples of a match, stored as yaml next to match.yaml."""

    def __init__(self, samples: list[EnrollmentSample] | None = None):
        self.samples = list(samples or [])

    @classmethod
    def load(cls, path: Path) -> Enrollment:
        """Samples from `path`; an empty enrollment when the file does not exist yet."""
        if not Path(path).exists():
            return cls()
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls([EnrollmentSample(s["name"], str(s["camera"]), int(s["frame"]),
                                     np.array(s["box"], dtype=np.float64))
                    for s in raw.get("samples") or []])

    def save(self, path: Path, fps: float | None = None) -> None:
        """Write the samples; with `fps` each one also gets a readable time (only informative)."""
        rows = []
        for s in sorted(self.samples, key=lambda s: (s.camera_id, s.frame, s.name)):
            row = {"name": s.name, "camera": s.camera_id, "frame": s.frame,
                   "box": [round(float(v), 1) for v in s.box]}
            if fps:
                row["time"] = format_time(s.frame / fps)
            rows.append(row)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump({"samples": rows}, sort_keys=False), encoding="utf-8")

    def add(self, sample: EnrollmentSample) -> None:
        """Add a sample, replacing an earlier one for the same player in the same frame."""
        self.samples = [s for s in self.samples
                        if not (s.name == sample.name and s.camera_id == sample.camera_id and s.frame == sample.frame)]
        self.samples.append(sample)

    def at_frame(self, camera_id: str, frame: int) -> list[EnrollmentSample]:
        return [s for s in self.samples if s.camera_id == camera_id and s.frame == frame]

    def remove(self, sample: EnrollmentSample) -> None:
        self.samples = [s for s in self.samples if s is not sample]

    def names(self) -> set[str]:
        return {s.name for s in self.samples}


def box_containing(boxes: np.ndarray, x: float, y: float) -> int | None:
    """Index of the smallest box containing pixel (x, y) - the front player when boxes overlap."""
    boxes = np.asarray(boxes).reshape(-1, 4)
    inside = (boxes[:, 0] <= x) & (x <= boxes[:, 2]) & (boxes[:, 1] <= y) & (y <= boxes[:, 3])
    if not inside.any():
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.flatnonzero(inside)[np.argmin(areas[inside])])
