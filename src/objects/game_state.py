"""Per-frame game state - the output contract of tracking, consumed by stats and visualization.

Saved as three CSV files in the run folder:
  players.csv: camera, frame, time_s, track, name, team, role, x1, y1, x2, y2, pitch_x, pitch_y, confidence
  ball.csv:    camera, frame, time_s, x1, y1, x2, y2, interpolated
  goals.csv:   camera, frame, x1, y1, x2, y2   (one row per goal / goalpost detection)
`name` is empty for players that could not be identified; pitch_x/pitch_y (metres) are empty
when the camera is not calibrated.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PLAYER_COLUMNS = ["camera", "frame", "time_s", "track", "name", "team", "role",
                  "x1", "y1", "x2", "y2", "pitch_x", "pitch_y", "confidence"]
BALL_COLUMNS = ["camera", "frame", "time_s", "x1", "y1", "x2", "y2", "interpolated"]
GOAL_COLUMNS = ["camera", "frame", "x1", "y1", "x2", "y2"]
TABLES = ("players", "ball", "goals")


@dataclass
class PlayerState:
    """One player in one frame."""
    track: str                  # tracklet key, stable while the online tracker kept the player
    name: str | None            # roster name, None = unknown
    team: str | None
    role: str                   # "player" or "goalkeeper"
    box: np.ndarray             # (4,) pixels
    pitch_xy: np.ndarray        # (2,) metres, NaN when uncalibrated
    confidence: float           # 0..1


@dataclass
class BallState:
    box: np.ndarray             # (4,) pixels
    interpolated: bool          # True when filled in between two detections


@dataclass
class FrameState:
    """Everything one camera saw in one frame."""
    camera: str
    frame: int
    time_s: float
    players: list[PlayerState] = field(default_factory=list)
    ball: BallState | None = None
    goals: list[np.ndarray] = field(default_factory=list)   # (4,) pixel boxes


class GameState:
    """Players and ball of every frame, stored as tables for fast per-player or per-frame queries."""

    def __init__(self, players: pd.DataFrame, ball: pd.DataFrame, goals: pd.DataFrame):
        self.players = players
        self.ball = ball
        self.goals = goals
        self._rows: dict[str, dict] = {}   # table -> {(camera, frame): row positions}, built on first lookup

    def save(self, folder: Path) -> None:
        Path(folder).mkdir(parents=True, exist_ok=True)
        for table in TABLES:
            getattr(self, table).to_csv(Path(folder) / f"{table}.csv", index=False)

    @classmethod
    def load(cls, folder: Path) -> GameState:
        # Only empty cells are missing values, so a player called "NA" stays a name.
        return cls(*(pd.read_csv(Path(folder) / f"{table}.csv", keep_default_na=False, na_values=[""],
                                 dtype={"camera": str, "track": str})
                     for table in TABLES))

    def frame(self, camera: str, frame: int) -> FrameState:
        """All objects of one camera frame (empty FrameState when nothing was seen)."""
        rows, ball_rows, goal_rows = (self._frame_rows(table, camera, frame) for table in TABLES)
        time_s = float(rows.time_s.iloc[0]) if len(rows) else np.nan
        state = FrameState(camera, frame, time_s, [_player_state(row) for row in rows.itertuples()])
        if len(ball_rows):
            row = ball_rows.iloc[0]
            state.ball = BallState(row[["x1", "y1", "x2", "y2"]].to_numpy(float), bool(row.interpolated))
        state.goals = list(goal_rows[["x1", "y1", "x2", "y2"]].to_numpy(float))
        return state

    def _frame_rows(self, table: str, camera: str, frame: int) -> pd.DataFrame:
        data = getattr(self, table)
        if table not in self._rows:
            self._rows[table] = data.groupby(["camera", "frame"]).indices if len(data) else {}
        return data.iloc[self._rows[table].get((camera, frame), [])]

    def names(self) -> list[str]:
        return sorted(self.players.name.dropna().unique())


def _player_state(row) -> PlayerState:
    def text(value):
        return None if pd.isna(value) else str(value)
    return PlayerState(str(row.track), text(row.name), text(row.team), str(row.role),
                       np.array([row.x1, row.y1, row.x2, row.y2], dtype=float),
                       np.array([row.pitch_x, row.pitch_y], dtype=float), float(row.confidence))
