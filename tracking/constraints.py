"""Hard rules deciding whether two tracklets may belong to the same player.

Each rule is its own class so camera setups can swap them: with one camera, time overlap always
means two different players; with two cameras, overlapping tracklets from different cameras are
the same player when their pitch positions agree.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from tracking.settings import TrackingSettings
from tracking.tracklet import Tracklet

PLAYER_HEIGHT_M = 1.8       # converts box heights to metres when a camera is not calibrated
POSITION_SLACK_M = 1.0      # detection jitter allowed on top of the running distance


def overlaps(a: Tracklet, b: Tracklet) -> bool:
    return a.start <= b.end and b.start <= a.end


class Constraint(ABC):
    @abstractmethod
    def compatible(self, a: Tracklet, b: Tracklet) -> bool:
        """False when `a` and `b` cannot be the same player."""


class NoOverlapSameCamera(Constraint):
    """One camera never sees the same player twice at the same moment."""

    def compatible(self, a: Tracklet, b: Tracklet) -> bool:
        return a.camera_id != b.camera_id or not overlaps(a, b)


class MaxSpeed(Constraint):
    """Between the end of one tracklet and the start of the next, the player can't run faster than
    `max_speed_mps`. Uses pitch metres, or box heights as a ruler on an uncalibrated camera."""

    def __init__(self, max_speed_mps: float):
        self.max_speed_mps = max_speed_mps

    def compatible(self, a: Tracklet, b: Tracklet) -> bool:
        """Allowed when the tracklets overlap in time, or the gap can be run at max speed."""
        if overlaps(a, b):
            return True
        first, second = (a, b) if a.end < b.start else (b, a)
        distance = self._distance_m(first, second)
        if np.isnan(distance):
            return True  # no way to measure (e.g. two uncalibrated cameras)
        gap = second.start - first.end
        return distance <= self.max_speed_mps * gap + POSITION_SLACK_M

    @staticmethod
    def _distance_m(first: Tracklet, second: Tracklet) -> float:
        start, end = first.pitch_xy[-1], second.pitch_xy[0]
        if not (np.isnan(start).any() or np.isnan(end).any()):
            return float(np.linalg.norm(end - start))
        if first.camera_id != second.camera_id:
            return np.nan
        box_a, box_b = first.boxes[-1], second.boxes[0]
        foot_a = np.array([(box_a[0] + box_a[2]) / 2, box_a[3]])
        foot_b = np.array([(box_b[0] + box_b[2]) / 2, box_b[3]])
        metres_per_pixel = PLAYER_HEIGHT_M / max((box_a[3] - box_a[1] + box_b[3] - box_b[1]) / 2, 1.0)
        return float(np.linalg.norm(foot_b - foot_a) * metres_per_pixel)


class CrossCameraAgreement(Constraint):
    """Two cameras may see one player at the same time only at the same pitch position."""

    def __init__(self, max_distance_m: float):
        self.max_distance_m = max_distance_m

    def compatible(self, a: Tracklet, b: Tracklet) -> bool:
        """Allowed for one camera or no time overlap; else positions must agree."""
        if a.camera_id == b.camera_id or not overlaps(a, b):
            return True
        times = a.times[(a.times >= b.start) & (a.times <= b.end)]
        b_xy = np.stack([np.interp(times, b.times, b.pitch_xy[:, i]) for i in range(2)], axis=1)
        a_xy = np.stack([np.interp(times, a.times, a.pitch_xy[:, i]) for i in range(2)], axis=1)
        distances = np.linalg.norm(a_xy - b_xy, axis=1)
        if np.isnan(distances).all():
            return False  # uncalibrated: can't confirm it is the same player -> treat as different
        return bool(np.nanmean(distances) <= self.max_distance_m)


def default_constraints(settings: TrackingSettings) -> list[Constraint]:
    return [NoOverlapSameCamera(), MaxSpeed(settings.max_speed_mps),
            CrossCameraAgreement(settings.cross_camera_max_dist_m)]
