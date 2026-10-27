"""Run detection + identity tracking for one match.

    py -3.13 -m tracking.pipeline matches/<game>/match.yaml [--max-frames N] [--refresh]

Writes players.csv / ball.csv / goals.csv (see src/objects/game_state.py) and review.md to the run folder.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from objects.game_state import GameState
from tracking.appearance import AppearanceEncoder, ColorHistogramEncoder
from tracking.camera import Camera
from tracking.config import MatchConfig
from tracking.enrollment import Enrollment
from tracking.gallery import PlayerGallery
from tracking.game_state_builder import GameStateBuilder
from tracking.identity import IdentityAssigner
from tracking.online_pass import PassCache, refresh_geometry
from tracking.online_tracker import (
    DEFAULT_TRACKER_CONFIG,
    OnlineTracker,
    YoloBotSortTracker,
)
from tracking.report import build_report
from tracking.timeline import Timeline


@dataclass
class PipelineResult:
    state: GameState
    report: str
    run_dir: Path


class TrackingPipeline:
    """Wires the stages together; tracker and encoder can be swapped (e.g. in tests)."""

    def __init__(self, config: MatchConfig, tracker: OnlineTracker | None = None,
                 encoder: AppearanceEncoder | None = None, tracker_config: Path = DEFAULT_TRACKER_CONFIG):
        self.config = config
        self.timeline = Timeline(config.timeline, config.roster)
        self.cameras = {c.id: Camera.from_config(c) for c in config.cameras}
        self.tracker_config = tracker_config
        self.tracker = tracker or YoloBotSortTracker(config.detector, tracker_config)
        self.encoder = encoder or ColorHistogramEncoder()

    def run(self, max_frames: int | None = None, refresh: bool = False) -> PipelineResult:
        """Run all stages; `max_frames` limits each video, `refresh` ignores the online-pass cache."""
        config, settings = self.config, self.config.settings
        cache = PassCache(config.run_dir)
        passes = []
        for camera in self.cameras.values():
            passes.append(cache.get_or_run(camera, self.tracker, self.encoder, settings, config.detector,
                                           self.tracker_config, max_frames, refresh))
            refresh_geometry(passes[-1].tracklets, camera)
        tracklets = [t for p in passes for t in p.tracklets]

        enrollment = Enrollment.load(config.enrollment)
        gallery = PlayerGallery.from_enrollment(enrollment, self.cameras, self.encoder, self.timeline)
        pieces, assignments = IdentityAssigner(self.timeline, settings).assign(tracklets, gallery, enrollment)

        builder = GameStateBuilder(self.timeline, config.max_players, settings.ball_max_gap_frames)
        state = builder.build(pieces, assignments, passes, self.cameras)
        state.save(config.run_dir)
        report = build_report(self.timeline, gallery, pieces, assignments, state, settings.report_min_unknown_s)
        (config.run_dir / "review.md").write_text(report, encoding="utf-8")
        return PipelineResult(state, report, config.run_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("match", type=Path, help="path to match.yaml")
    parser.add_argument("--max-frames", type=int, help="only process the first N frames of each camera")
    parser.add_argument("--refresh", action="store_true", help="ignore the cached online pass")
    args = parser.parse_args()
    result = TrackingPipeline(MatchConfig.load(args.match)).run(args.max_frames, args.refresh)
    print(result.report)
    print(f"Results in {result.run_dir}")


if __name__ == "__main__":
    main()
