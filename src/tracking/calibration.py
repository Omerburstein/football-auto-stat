"""Image -> pitch mapping of a fixed camera, from points the user clicked (src/tools/calibrate.py).

Pitch coordinates are metres: x along the length (0 = left goal line), y along the width
(0 = the touchline at the top of the standard top-down drawing).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from tracking.config import PitchConfig

MIN_POINTS = 4  # a homography has 8 degrees of freedom -> 4 point pairs


def pitch_landmarks(pitch: PitchConfig) -> dict[str, tuple[float, float]]:
    """Named pitch points (metres) that exist on any pitch size, for clicking during calibration."""
    length, width = pitch.length_m, pitch.width_m
    return {
        "corner_top_left": (0.0, 0.0),
        "corner_top_right": (length, 0.0),
        "corner_bottom_left": (0.0, width),
        "corner_bottom_right": (length, width),
        "halfway_top": (length / 2, 0.0),
        "halfway_bottom": (length / 2, width),
        "center_spot": (length / 2, width / 2),
        "goal_left_center": (0.0, width / 2),
        "goal_right_center": (length, width / 2),
    }


def _collinear(points: np.ndarray, tolerance: float = 1e-3) -> bool:
    """True when all points lie (almost) on one line - a homography can't be fitted from them."""
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return singular[0] == 0 or singular[-1] / singular[0] < tolerance


@dataclass
class Calibration:
    """Homography from image pixels to pitch metres plus the points it was fitted on."""
    image_points: np.ndarray   # (N, 2) pixels
    pitch_points: np.ndarray   # (N, 2) metres
    homography: np.ndarray     # (3, 3)

    @classmethod
    def fit(cls, image_points, pitch_points) -> Calibration:
        """Least-squares homography; raises ValueError for < 4 points or a degenerate (collinear) set."""
        image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        pitch_points = np.asarray(pitch_points, dtype=np.float64).reshape(-1, 2)
        if len(image_points) < MIN_POINTS or len(image_points) != len(pitch_points):
            raise ValueError(f"need >= {MIN_POINTS} matching image/pitch points")
        homography = None
        if not (_collinear(image_points) or _collinear(pitch_points)):
            homography, _ = cv2.findHomography(image_points, pitch_points, method=0)
        if homography is None:
            raise ValueError("points are degenerate (e.g. collinear) - click points spread over the pitch")
        return cls(image_points, pitch_points, homography)

    def image_to_pitch(self, points) -> np.ndarray:
        """(N, 2) pixels -> (N, 2) metres."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        if len(points) == 0:
            return np.empty((0, 2))
        return cv2.perspectiveTransform(points, self.homography).reshape(-1, 2)

    @property
    def reprojection_error_m(self) -> float:
        """Mean distance (m) between the clicked pitch points and where the homography maps them."""
        mapped = self.image_to_pitch(self.image_points)
        return float(np.linalg.norm(mapped - self.pitch_points, axis=1).mean())

    def save(self, path: Path) -> None:
        """Write points, homography and reprojection error as yaml."""
        data = {
            "image_points": self.image_points.tolist(),
            "pitch_points": self.pitch_points.tolist(),
            "homography": self.homography.tolist(),
            "reprojection_error_m": round(self.reprojection_error_m, 3),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Calibration:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(np.array(data["image_points"], dtype=np.float64),
                   np.array(data["pitch_points"], dtype=np.float64),
                   np.array(data["homography"], dtype=np.float64))
