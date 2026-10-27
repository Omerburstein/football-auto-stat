"""match.yaml - the single file where the user enters every assumption about a game.

Relative paths inside match.yaml are resolved against the folder that contains it.
See matches/example/match.yaml for a documented example.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from tracking.settings import TrackingSettings

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "weights" / "best_both.pt"


class ConfigError(ValueError):
    """match.yaml is missing data or contradicts itself."""


def parse_time(value) -> float:
    """Seconds from a number or a 'mm:ss' / 'hh:mm:ss' string (fractions allowed: '1:02.5')."""
    if isinstance(value, (int, float)):
        return float(value)
    seconds = 0.0
    for part in str(value).split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def format_time(seconds: float) -> str:
    """'mm:ss.s' for reports and tool prompts."""
    minutes, rest = divmod(max(seconds, 0.0), 60)
    return f"{int(minutes):02d}:{rest:04.1f}"


def _as_list(value) -> list[str]:
    if value is None:
        return []
    return [value] if isinstance(value, str) else list(value)


@dataclass
class PitchConfig:
    length_m: float
    width_m: float


@dataclass
class CameraConfig:
    id: str
    video: Path
    calibration: Path          # written by src/tools/calibrate.py; may not exist yet
    time_offset_s: float = 0.0  # added to this camera's frame times to sync several cameras


@dataclass
class DetectorConfig:
    """Which model finds players, ball and goals, and its class names."""
    weights: Path
    conf: float = 0.25
    imgsz: int = 960     # model input width; 640 misses far-away players in 1080p phone video
    player_label: str = "player"
    ball_label: str = "ball"
    goal_label: str = "goalpost"


@dataclass
class TimelineEvent:
    """A change during the game, effective from `t` seconds onwards."""
    t: float
    teams: dict[str, list[str]] | None = None   # full new line-up: team -> players on the field
    moves: dict[str, str] = field(default_factory=dict)        # player -> new team
    goalkeepers: dict[str, str] = field(default_factory=dict)  # team -> goalkeeper
    outfit_change: list[str] = field(default_factory=list)
    sub_in: list[str] = field(default_factory=list)
    sub_out: list[str] = field(default_factory=list)

    KEYS = ("t", "teams", "moves", "goalkeepers", "outfit_change", "sub_in", "sub_out")

    @classmethod
    def from_dict(cls, raw: dict) -> TimelineEvent:
        """Parse one yaml timeline entry; ConfigError on unknown keys or a missing 't'."""
        unknown = set(raw) - set(cls.KEYS)
        if unknown or "t" not in raw:
            raise ConfigError(f"Timeline event {raw}: needs 't', unknown keys {sorted(unknown)}")
        teams = raw.get("teams")
        return cls(
            t=parse_time(raw["t"]),
            teams={team: _as_list(names) for team, names in teams.items()} if teams else None,
            moves=dict(raw.get("moves") or {}),
            goalkeepers=dict(raw.get("goalkeepers") or {}),
            outfit_change=_as_list(raw.get("outfit_change")),
            sub_in=_as_list(raw.get("sub_in")),
            sub_out=_as_list(raw.get("sub_out")),
        )

    def names(self) -> set[str]:
        """Every player name this event mentions (for roster validation)."""
        names = set(self.moves) | set(self.goalkeepers.values())
        names |= set(self.outfit_change) | set(self.sub_in) | set(self.sub_out)
        for team_names in (self.teams or {}).values():
            names |= set(team_names)
        return names


@dataclass
class MatchConfig:
    """All inputs of one match, with paths already resolved."""
    path: Path
    max_players: int
    pitch: PitchConfig
    cameras: list[CameraConfig]
    roster: list[str]
    enrollment: Path
    timeline: list[TimelineEvent]
    detector: DetectorConfig
    settings: TrackingSettings
    run_dir: Path               # cached stage outputs and results

    @classmethod
    def load(cls, path: str | Path) -> MatchConfig:
        """Read and validate match.yaml; ConfigError when something is missing or contradictory."""
        path = Path(path).resolve()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        base = path.parent
        try:
            config = cls(
                path=path,
                max_players=int(raw["max_players"]),
                pitch=PitchConfig(**raw["pitch"]),
                cameras=[_camera(c, base) for c in raw["cameras"]],
                roster=_as_list(raw["roster"]),
                enrollment=base / raw.get("enrollment", "enrollment.yaml"),
                timeline=[TimelineEvent.from_dict(e) for e in raw.get("timeline") or []],
                detector=_detector(raw.get("detector") or {}, base),
                settings=TrackingSettings.from_dict(raw.get("tracking")),
                run_dir=base / raw.get("run_dir", "run"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ConfigError(f"{path}: {error!r}") from error
        config.validate()
        return config

    def validate(self) -> None:
        """Raise ConfigError on contradictions the pipeline cannot recover from."""
        if self.max_players <= 0:
            raise ConfigError("max_players must be positive")
        if len(set(self.roster)) != len(self.roster):
            raise ConfigError("roster has duplicate names")
        if not self.cameras or len({c.id for c in self.cameras}) != len(self.cameras):
            raise ConfigError("need at least one camera, with unique ids")
        for event in self.timeline:
            unknown = event.names() - set(self.roster)
            if unknown:
                raise ConfigError(f"timeline at {format_time(event.t)} names players not in roster: {sorted(unknown)}")

    def camera(self, camera_id: str) -> CameraConfig:
        return next(c for c in self.cameras if c.id == camera_id)


def _camera(raw: dict, base: Path) -> CameraConfig:
    return CameraConfig(
        id=str(raw["id"]),
        video=base / raw["video"],
        calibration=base / raw.get("calibration", f"{raw['id']}_calibration.yaml"),
        time_offset_s=float(raw.get("time_offset_s", 0.0)),
    )


def _detector(raw: dict, base: Path) -> DetectorConfig:
    raw = dict(raw)
    weights = base / raw.pop("weights") if "weights" in raw else DEFAULT_WEIGHTS
    return DetectorConfig(weights=weights, **raw)
