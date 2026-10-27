import numpy as np
import pandas as pd

from objects.game_state import BALL_COLUMNS, GOAL_COLUMNS, GameState
from tests.helpers import OUTFITS, make_tracklet
from tracking.config import TimelineEvent
from tracking.gallery import PlayerGallery
from tracking.game_state_builder import GameStateBuilder
from tracking.identity import Assignment
from tracking.online_pass import CameraPass
from tracking.report import build_report
from tracking.timeline import GOALKEEPER, Timeline


class FakeCamera:
    def frame_time(self, frames):
        return np.asarray(frames) / 10.0


TIMELINE = Timeline([TimelineEvent.from_dict({"t": 0, "teams": {"A": ["Dan"], "B": ["Tal"]},
                                              "goalkeepers": {"B": "Tal"}})], ["Dan", "Tal", "Gil"])


def camera_pass(ball_frames=(), frame_count=10, goals=None) -> CameraPass:
    ball = {f: np.array([f, 0, f + 2, 2.0]) for f in ball_frames}
    return CameraPass("cam1", frame_count, [], ball, goals or {})


def build(tracklets, assignments, max_players=5, ball_frames=(), max_gap=3) -> GameState:
    builder = GameStateBuilder(TIMELINE, max_players, max_gap)
    return builder.build(tracklets, assignments, [camera_pass(ball_frames)], {"cam1": FakeCamera()})


def test_player_rows_carry_name_team_role_and_confidence():
    tracklets = [make_tracklet("cam1:1", 0, 5), make_tracklet("cam1:2", 0, 5, x_start=5), make_tracklet("cam1:3", 0, 5)]
    assignments = {"cam1:1": Assignment("Dan", 0.1), "cam1:2": Assignment("Tal", 0.0, anchored=True),
                   "cam1:3": Assignment(None)}
    players = build(tracklets, assignments).players
    dan = players[players.name == "Dan"].iloc[0]
    assert (dan.team, dan.role, dan.confidence) == ("A", "player", 0.9)
    assert players[players.name == "Tal"].iloc[0].role == GOALKEEPER
    assert players[players.track == "cam1:3"].name.isna().all()
    assert len(players) == 15


def test_extra_people_beyond_max_players_are_dropped_unknown_first():
    tracklets = [make_tracklet("cam1:1", 0, 5), make_tracklet("cam1:2", 0, 5), make_tracklet("cam1:3", 0, 3)]
    assignments = {"cam1:1": Assignment(None), "cam1:2": Assignment("Dan", 0.1), "cam1:3": Assignment(None)}
    players = build(tracklets, assignments, max_players=2).players
    assert players.groupby("frame").size().max() == 2
    assert set(players.track) == {"cam1:1", "cam1:2"}     # the shorter unknown track is dropped


def test_ball_short_gaps_are_interpolated_long_gaps_left_empty():
    ball = build([], {}, ball_frames=(0, 3, 9), max_gap=2).ball
    assert ball.frame.tolist() == [0, 1, 2, 3, 9]
    assert ball.interpolated.tolist() == [False, True, True, False, False]
    assert np.isclose(ball.set_index("frame").loc[1, "x1"], 1.0)


def test_game_state_round_trip_and_frame_lookup(tmp_path):
    tracklets = [make_tracklet("cam1:1", 0, 5)]
    builder = GameStateBuilder(TIMELINE, 5, 3)
    goals = {2: np.array([[1, 2, 3, 4.0]])}
    state = builder.build(tracklets, {"cam1:1": Assignment("Dan", 0.2)},
                          [CameraPass("cam1", 5, [], {2: np.array([0, 0, 2, 2.0])}, goals)], {"cam1": FakeCamera()})
    state.save(tmp_path)
    loaded = GameState.load(tmp_path)
    frame = loaded.frame("cam1", 2)
    assert [p.name for p in frame.players] == ["Dan"]
    assert frame.players[0].team == "A"
    assert frame.ball is not None and not frame.ball.interpolated
    assert np.allclose(frame.goals[0], [1, 2, 3, 4])
    assert loaded.frame("cam1", 99).players == []
    assert loaded.names() == ["Dan"]


def test_unknown_names_survive_the_csv_round_trip(tmp_path):
    state = GameState(pd.DataFrame({"camera": ["cam1"], "frame": [0], "time_s": [0.0], "track": ["1"], "name": [None],
                                    "team": [None], "role": ["player"], "x1": [0], "y1": [0], "x2": [1], "y2": [1],
                                    "pitch_x": [np.nan], "pitch_y": [np.nan], "confidence": [0.0]}),
                      pd.DataFrame(columns=BALL_COLUMNS), pd.DataFrame(columns=GOAL_COLUMNS))
    state.save(tmp_path)
    player = GameState.load(tmp_path).frame("cam1", 0).players[0]
    assert player.name is None and player.track == "1"


def test_report_lists_missing_outfits_and_unknown_tracklets():
    gallery = PlayerGallery()
    gallery.add("Dan", 0, OUTFITS["red"])
    tracklets = [make_tracklet("cam1:1", 0, 30), make_tracklet("cam1:2", 0, 30)]
    assignments = {"cam1:1": Assignment("Dan", 0.1), "cam1:2": Assignment(None)}
    state = build(tracklets, assignments)
    report = build_report(TIMELINE, gallery, tracklets, assignments, state, min_unknown_s=1.0)
    assert "| Dan | 2.9 | 1 |" in report
    assert "- Tal: outfit #0" in report and "- Gil: outfit #0" in report
    assert "| cam1 | cam1:2 |" in report


def test_report_says_when_nobody_is_enrolled():
    report = build_report(TIMELINE, PlayerGallery(), [], {}, build([], {}), min_unknown_s=1.0)
    assert "No players enrolled yet" in report
