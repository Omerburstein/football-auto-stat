"""Offline global identity assignment: which roster player each tracklet belongs to.

1. Split tracklets at likely id swaps (appearance jumps, conflicting enrollment clicks).
2. Pin tracklets that contain an enrollment click to that name.
3. Longest tracklets first: take everything active at that moment and match it to players with the
   Hungarian algorithm on appearance distance, skipping players ruled out by the hard constraints
   (already seen elsewhere at that time, too far away to run there, not on the field).
4. Add confident matches to the gallery and solve again, so later views of a player are recognised.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.constraints import Constraint, default_constraints
from tracking.enrollment import Enrollment
from tracking.gallery import PlayerGallery
from tracking.settings import TrackingSettings
from tracking.splitting import anchors_of, split_by_anchors, split_by_appearance
from tracking.timeline import Timeline
from tracking.tracklet import Tracklet

UNASSIGNABLE = 1e6  # stands in for inf inside the Hungarian solver


@dataclass
class Assignment:
    """Who a tracklet is, and how sure the assignment is."""
    name: str | None          # None = unknown player
    distance: float = np.inf  # appearance distance to the player's gallery
    anchored: bool = False    # pinned by an enrollment click

    @property
    def confidence(self) -> float:
        """1 for enrollment-pinned tracklets, 1 - distance otherwise, 0 for unknown."""
        if self.anchored:
            return 1.0
        return 0.0 if self.name is None else float(np.clip(1.0 - self.distance, 0.0, 1.0))


class IdentityAssigner:
    """Names every tracklet with a roster player (or None) under the hard constraints."""

    def __init__(self, timeline: Timeline, settings: TrackingSettings, constraints: list[Constraint] | None = None):
        self.timeline = timeline
        self.settings = settings
        self.constraints = constraints if constraints is not None else default_constraints(settings)

    def assign(self, tracklets: list[Tracklet], gallery: PlayerGallery,
               enrollment: Enrollment) -> tuple[list[Tracklet], dict[str, Assignment]]:
        """Returns the (possibly split) tracklets and an Assignment per tracklet key."""
        pieces, anchors = self.prepare(tracklets, enrollment)
        round_gallery = gallery
        for _ in range(self.settings.refinement_rounds + 1):
            assignments = self._solve(pieces, round_gallery, anchors)
            round_gallery = self._grown_gallery(gallery, pieces, assignments)
        return pieces, assignments

    def prepare(self, tracklets: list[Tracklet], enrollment: Enrollment) -> tuple[list[Tracklet], dict[str, str]]:
        """Split tracklets and find which pieces enrollment clicks pin to a name."""
        pieces, anchors = [], {}
        for tracklet in tracklets:
            for part in split_by_appearance(tracklet, self.settings):
                for piece, name in split_by_anchors(part, anchors_of(part, enrollment, self.settings.anchor_iou)):
                    pieces.append(piece)
                    if name is not None:
                        anchors[piece.key] = name
        return pieces, anchors

    def _solve(self, pieces: list[Tracklet], gallery: PlayerGallery, anchors: dict[str, str]) -> dict[str, Assignment]:
        roster = self.timeline.roster
        assignments = {t.key: Assignment(None) for t in pieces}
        owned: dict[str, list[Tracklet]] = defaultdict(list)
        for piece in pieces:
            if piece.key in anchors:
                assignments[piece.key] = Assignment(anchors[piece.key], 0.0, anchored=True)
                owned[anchors[piece.key]].append(piece)

        pending = sorted((t for t in pieces if t.key not in anchors), key=lambda t: -t.duration)
        while pending:
            moment = (pending[0].start + pending[0].end) / 2
            group = [t for t in pending if t.start <= moment <= t.end]
            costs = np.array([[self._cost(t, name, gallery, owned) for name in roster] for t in group])
            rows, cols = linear_sum_assignment(np.where(np.isfinite(costs), costs, UNASSIGNABLE))
            for row, col in zip(rows, cols):
                if costs[row, col] <= self.settings.max_assign_distance:
                    assignments[group[row].key] = Assignment(roster[col], float(costs[row, col]))
                    owned[roster[col]].append(group[row])
            grouped = {t.key for t in group}
            pending = [t for t in pending if t.key not in grouped]
        return assignments

    def _cost(self, tracklet: Tracklet, name: str, gallery: PlayerGallery, owned: dict[str, list[Tracklet]]) -> float:
        """Appearance distance of `tracklet` to `name`, inf when a hard rule forbids the match."""
        middle = (tracklet.start + tracklet.end) / 2
        prototype = tracklet.prototype
        if prototype is None or not self.timeline.is_on_field(name, middle):
            return np.inf
        if not all(c.compatible(tracklet, other) for other in owned[name] for c in self.constraints):
            return np.inf
        return gallery.distance(name, self.timeline.outfit_period(name, middle), prototype)

    def _grown_gallery(self, base: PlayerGallery, pieces: list[Tracklet],
                       assignments: dict[str, Assignment]) -> PlayerGallery:
        """The enrolled gallery plus the mean outfit of every confidently identified tracklet."""
        grown = base.copy()
        for piece in pieces:
            assignment = assignments[piece.key]
            confident = assignment.anchored or assignment.distance <= self.settings.confident_distance
            if assignment.name is not None and confident and piece.prototype is not None:
                period = self.timeline.outfit_period(assignment.name, (piece.start + piece.end) / 2)
                grown.add(assignment.name, period, piece.prototype)
        return grown
