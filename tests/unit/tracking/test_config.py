from pathlib import Path

import pytest
import yaml

from tracking.config import (
    ConfigError,
    MatchConfig,
    TimelineEvent,
    format_time,
    parse_time,
)
from tracking.settings import TrackingSettings

EXAMPLE = Path(__file__).resolve().parents[3] / "matches" / "example" / "match.yaml"


def write_match(tmp_path, **overrides) -> Path:
    match = {"max_players": 4, "pitch": {"length_m": 40, "width_m": 20},
             "cameras": [{"id": "cam1", "video": "v.mp4"}], "roster": ["A", "B", "C"]}
    match.update(overrides)
    path = tmp_path / "match.yaml"
    path.write_text(yaml.safe_dump(match), encoding="utf-8")
    return path


@pytest.mark.parametrize("value, seconds", [(12, 12.0), (1.5, 1.5), ("12:30", 750.0), ("1:00:05", 3605.0),
                                            ("0:02.5", 2.5), ("45", 45.0)])
def test_parse_time_accepts_numbers_and_clock_strings(value, seconds):
    assert parse_time(value) == seconds


def test_format_time_is_minutes_and_seconds():
    assert format_time(750.0) == "12:30.0"
    assert format_time(-3) == "00:00.0"


def test_example_match_loads_with_paths_relative_to_the_yaml():
    config = MatchConfig.load(EXAMPLE)
    assert config.max_players == 10
    assert config.cameras[0].video.parent.name == "videos"
    assert config.cameras[0].calibration == EXAMPLE.parent / "cam1_calibration.yaml"
    assert config.enrollment == EXAMPLE.parent / "enrollment.yaml"
    assert config.run_dir == EXAMPLE.parent / "run"
    assert config.detector.weights.name == "best_both.pt"
    assert len(config.timeline) == 4


def test_defaults_when_optional_sections_are_missing(tmp_path):
    config = MatchConfig.load(write_match(tmp_path))
    assert config.timeline == []
    assert config.settings == TrackingSettings()
    assert config.cameras[0].time_offset_s == 0.0


def test_tracking_section_overrides_settings(tmp_path):
    config = MatchConfig.load(write_match(tmp_path, tracking={"max_speed_mps": 9.5}))
    assert config.settings.max_speed_mps == 9.5


@pytest.mark.parametrize("overrides", [
    {"tracking": {"max_sped_mps": 9}},                           # typo in a setting
    {"timeline": [{"t": 0, "goalkeeper": {"A": "A"}}]},          # typo in an event key
    {"timeline": [{"teams": {"X": ["A"]}}]},                     # event without time
    {"timeline": [{"t": 0, "sub_in": ["Nobody"]}]},              # name not in roster
    {"roster": ["A", "A"]},
    {"max_players": 0},
    {"cameras": [{"id": "c", "video": "a.mp4"}, {"id": "c", "video": "b.mp4"}]},
])
def test_invalid_match_raises_config_error(tmp_path, overrides):
    with pytest.raises(ConfigError):
        MatchConfig.load(write_match(tmp_path, **overrides))


def test_missing_required_key_raises_config_error(tmp_path):
    path = write_match(tmp_path)
    raw = yaml.safe_load(path.read_text())
    del raw["pitch"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError):
        MatchConfig.load(path)


def test_timeline_event_accepts_single_names_and_lists():
    event = TimelineEvent.from_dict({"t": "1:00", "outfit_change": "Dan", "sub_in": ["A", "B"]})
    assert event.t == 60.0
    assert event.outfit_change == ["Dan"]
    assert event.names() == {"Dan", "A", "B"}
