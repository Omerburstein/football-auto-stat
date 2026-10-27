"""Tunable thresholds of the tracking pipeline.

Every field can be overridden from the optional `tracking:` section of match.yaml, e.g.
`tracking: {max_speed_mps: 9}`. Appearance distances are cosine distances (0 = identical, 1 = unrelated).
"""
from dataclasses import dataclass, fields


@dataclass
class TrackingSettings:
    """Thresholds of every stage; defaults suit a phone filming a small pitch."""

    # --- appearance sampling during the online pass ---
    embed_every_frames: int = 5         # one appearance sample per track every N frames
    min_box_height_px: int = 25         # smaller boxes are too blurry to describe an outfit
    max_overlap_iou: float = 0.05       # crops overlapping another player more than this mix two outfits
    min_tracklet_frames: int = 5        # shorter tracklets are mostly false detections

    # --- offline identity assignment ---
    split_distance: float = 0.30        # appearance jump inside one tracklet that means an id switch
    min_split_samples: int = 3          # samples needed on each side of a split
    max_assign_distance: float = 0.40   # farther than this from every player -> stays "unknown"
    confident_distance: float = 0.20    # assignments this close feed the gallery in refinement rounds
    refinement_rounds: int = 2
    anchor_iou: float = 0.5             # enrollment box vs tracklet box overlap that pins the tracklet
    max_speed_mps: float = 8.0          # faster "jumps" between two tracklets can't be one player
    cross_camera_max_dist_m: float = 1.5  # two cameras see the same player within this distance

    # --- output ---
    ball_max_gap_frames: int = 25       # longer ball dropouts are left empty instead of interpolated
    report_min_unknown_s: float = 1.0   # shorter unknown tracklets are not listed in the review report

    @classmethod
    def from_dict(cls, raw: dict | None) -> "TrackingSettings":
        """Build settings from the yaml section; unknown keys raise ValueError (catches typos)."""
        raw = raw or {}
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown tracking settings: {sorted(unknown)}")
        return cls(**raw)
