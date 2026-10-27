"""Turns identified tracklets plus ball and goal detections into the per-frame GameState tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from objects.game_state import BALL_COLUMNS, GOAL_COLUMNS, PLAYER_COLUMNS, GameState
from tracking.camera import Camera
from tracking.identity import Assignment
from tracking.online_pass import CameraPass
from tracking.timeline import FIELD_PLAYER, Timeline
from tracking.tracklet import Tracklet

BOX = ["x1", "y1", "x2", "y2"]


class GameStateBuilder:
    """Combines tracklets, their assigned names and the timeline into GameState tables."""

    def __init__(self, timeline: Timeline, max_players: int, ball_max_gap_frames: int):
        self.timeline = timeline
        self.max_players = max_players
        self.ball_max_gap_frames = ball_max_gap_frames

    def build(self, tracklets: list[Tracklet], assignments: dict[str, Assignment],
              passes: list[CameraPass], cameras: dict[str, Camera]) -> GameState:
        """GameState with one player row per tracklet frame, ball and goal rows per camera."""
        players = pd.concat([self._tracklet_rows(t, assignments[t.key]) for t in tracklets] or
                            [pd.DataFrame(columns=PLAYER_COLUMNS)], ignore_index=True)
        balls = [self._ball_rows(p, cameras[p.camera_id]) for p in passes]
        goals = [self._goal_rows(p) for p in passes]
        return GameState(self._limit_players(players),
                         pd.concat(balls, ignore_index=True) if balls else pd.DataFrame(columns=BALL_COLUMNS),
                         pd.concat(goals, ignore_index=True) if goals else pd.DataFrame(columns=GOAL_COLUMNS))

    def _tracklet_rows(self, tracklet: Tracklet, assignment: Assignment) -> pd.DataFrame:
        name = assignment.name
        rows = pd.DataFrame(tracklet.boxes, columns=BOX)
        rows.insert(0, "camera", tracklet.camera_id)
        rows.insert(1, "frame", tracklet.frames)
        rows.insert(2, "time_s", tracklet.times)
        rows.insert(3, "track", tracklet.key)
        rows.insert(4, "name", name)
        rows.insert(5, "team", [self.timeline.team_of(name, t) for t in tracklet.times] if name else None)
        rows.insert(6, "role", [self.timeline.role_of(name, t) for t in tracklet.times] if name else FIELD_PLAYER)
        rows["pitch_x"], rows["pitch_y"] = tracklet.pitch_xy[:, 0], tracklet.pitch_xy[:, 1]
        rows["confidence"] = assignment.confidence
        rows["_duration"] = tracklet.duration
        return rows

    def _limit_players(self, players: pd.DataFrame) -> pd.DataFrame:
        """Keep at most `max_players` per frame: identified players first, then the longest unknown
        tracks. The extras are referees, coaches and spectators the detector also calls 'player'."""
        if players.empty:
            return players[PLAYER_COLUMNS]
        players = players.assign(_named=players.name.notna())
        players = players.sort_values(["camera", "frame", "_named", "_duration"], ascending=[True, True, False, False])
        players = players.groupby(["camera", "frame"], sort=False).head(self.max_players)
        return players[PLAYER_COLUMNS].sort_values(["camera", "frame", "track"]).reset_index(drop=True)

    def _ball_rows(self, camera_pass: CameraPass, camera: Camera) -> pd.DataFrame:
        """Ball per frame; gaps up to `ball_max_gap_frames` are filled linearly and flagged."""
        boxes = pd.DataFrame(np.nan, index=pd.RangeIndex(camera_pass.frame_count, name="frame"), columns=BOX)
        for frame, box in camera_pass.ball_boxes.items():
            boxes.loc[frame] = box
        missing = boxes.x1.isna()
        run_length = missing.groupby((missing != missing.shift()).cumsum()).transform("size")
        filled = boxes.interpolate(limit_area="inside")
        filled[missing & (run_length > self.ball_max_gap_frames)] = np.nan
        filled = filled.dropna().reset_index()
        filled.insert(0, "camera", camera_pass.camera_id)
        filled.insert(2, "time_s", camera.frame_time(filled.frame.to_numpy()))
        filled["interpolated"] = missing.loc[filled.frame].to_numpy()
        return filled[BALL_COLUMNS]

    @staticmethod
    def _goal_rows(camera_pass: CameraPass) -> pd.DataFrame:
        rows = [[camera_pass.camera_id, frame, *box] for frame, boxes in sorted(camera_pass.goal_boxes.items())
                for box in boxes]
        return pd.DataFrame(rows, columns=GOAL_COLUMNS)
