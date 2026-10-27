"""The whole pipeline on a synthetic match (real video file, fake online tracker), plus a real-model smoke test."""
from pathlib import Path

import numpy as np
import pytest

from objects.game_state import GameState
from tests.helpers import BREAK_FRAME, SCENE_FRAMES, SceneTracker, scene_boxes
from tracking.config import MatchConfig
from tracking.geometry import iou_matrix
from tracking.pipeline import TrackingPipeline
from visualization.render_video import render

ROOT = Path(__file__).resolve().parents[2]


def true_name(row) -> str:
    """Name of the scene player whose box this output row is."""
    boxes = scene_boxes(int(row.frame))
    box = np.array([row.x1, row.y1, row.x2, row.y2])
    return max(boxes, key=lambda name: iou_matrix(boxes[name], box)[0, 0])


def test_names_survive_a_tracker_break_and_teams_come_from_the_timeline(scene_match):
    result = TrackingPipeline(MatchConfig.load(scene_match), tracker=SceneTracker()).run()
    players = result.state.players
    assert len(players) == 3 * SCENE_FRAMES
    assert players.name.notna().all()
    assert all(row.name == true_name(row) for row in players.itertuples())
    after_break = players[players.frame >= BREAK_FRAME]
    assert after_break.track.str.startswith("cam1:1").all()        # new online ids 11-13
    assert set(players[players.name == "Blue"].role) == {"goalkeeper"}
    assert set(players[players.name == "Red"].team) == {"A"}


def test_outputs_are_written_and_reloadable(scene_match):
    result = TrackingPipeline(MatchConfig.load(scene_match), tracker=SceneTracker()).run()
    for name in ("players.csv", "ball.csv", "goals.csv", "review.md", "online_cam1.pkl"):
        assert (result.run_dir / name).exists()
    assert GameState.load(result.run_dir).names() == ["Blue", "Red", "Yellow"]


def test_second_run_reuses_the_cached_online_pass(scene_match):
    config = MatchConfig.load(scene_match)
    TrackingPipeline(config, tracker=SceneTracker()).run()

    class FailingTracker(SceneTracker):
        def run(self, camera, max_frames=None):
            raise AssertionError("online pass should come from the cache")

    TrackingPipeline(config, tracker=FailingTracker()).run()


def test_overlay_video_has_every_frame(scene_match, tmp_path):
    pipeline = TrackingPipeline(MatchConfig.load(scene_match), tracker=SceneTracker())
    result = pipeline.run()
    assert render(pipeline.cameras["cam1"], result.state, tmp_path / "out.mp4") == SCENE_FRAMES


@pytest.mark.slow
def test_real_model_on_the_first_frames_of_the_test_video(tmp_path):
    match = ROOT / "matches" / "test_video" / "match.yaml"
    if not (ROOT / "weights" / "best_both.pt").exists() or not any((ROOT / "tests_videos").glob("*.mp4")):
        pytest.skip("needs weights/best_both.pt and tests_videos/")
    config = MatchConfig.load(match)
    config.run_dir = tmp_path
    config.enrollment = tmp_path / "none.yaml"
    result = TrackingPipeline(config).run(max_frames=15)
    players = result.state.players
    assert players.frame.max() <= 14
    assert players.groupby("frame").size().max() <= config.max_players
    assert players.name.isna().all()            # nobody enrolled -> nobody named
    assert "No players enrolled yet" in result.report
