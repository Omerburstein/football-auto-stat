import pytest

from tracking.config import ConfigError, TimelineEvent
from tracking.timeline import FIELD_PLAYER, GOALKEEPER, Timeline

ROSTER = ["Amir", "Dan", "Tal", "Gil", "Maya"]


def timeline(*events) -> Timeline:
    return Timeline([TimelineEvent.from_dict(e) for e in events], ROSTER)


START = {"t": 0, "teams": {"A": ["Amir", "Dan"], "B": ["Tal", "Gil"]}, "goalkeepers": {"A": "Dan", "B": "Tal"}}


def test_without_events_everyone_is_on_field_without_team():
    empty = timeline()
    assert empty.is_on_field("Maya", 100)
    assert empty.team_of("Amir", 0) is None
    assert empty.role_of("Amir", 0) == FIELD_PLAYER


def test_line_up_sets_teams_roles_and_who_plays():
    line_up = timeline(START)
    assert line_up.team_of("Amir", 0) == "A"
    assert line_up.role_of("Dan", 0) == GOALKEEPER
    assert line_up.role_of("Amir", 0) == FIELD_PLAYER
    assert not line_up.is_on_field("Maya", 0)


def test_goalkeeper_change_applies_from_its_time_only():
    changed = timeline(START, {"t": "10:00", "goalkeepers": {"A": "Amir"}})
    assert changed.role_of("Dan", 599) == GOALKEEPER
    assert changed.role_of("Dan", 600) == FIELD_PLAYER
    assert changed.role_of("Amir", 600) == GOALKEEPER


def test_move_to_other_team_drops_goalkeeper_role():
    moved = timeline(START, {"t": 60, "moves": {"Dan": "B"}})
    assert moved.team_of("Dan", 60) == "B"
    assert moved.role_of("Dan", 60) == FIELD_PLAYER
    assert moved.role_of("Tal", 60) == GOALKEEPER


def test_substitution_changes_who_is_on_field():
    subbed = timeline(START, {"t": 60, "sub_out": ["Gil"], "sub_in": ["Maya"], "moves": {"Maya": "B"}})
    assert not subbed.is_on_field("Gil", 60)
    assert subbed.is_on_field("Maya", 60)
    assert subbed.team_of("Maya", 60) == "B"


def test_events_out_of_order_are_sorted():
    shuffled = timeline({"t": 60, "goalkeepers": {"A": "Amir"}}, START)
    assert shuffled.role_of("Amir", 70) == GOALKEEPER


def test_outfit_periods_count_changes_per_player():
    outfits = timeline(START, {"t": 100, "outfit_change": ["Dan"]}, {"t": 200, "outfit_change": ["Dan", "Tal"]})
    assert [outfits.outfit_period("Dan", t) for t in (0, 100, 199, 200)] == [0, 1, 1, 2]
    assert outfits.outfit_period("Tal", 150) == 0
    assert outfits.outfit_starts("Dan") == [0.0, 100.0, 200.0]
    assert outfits.outfit_starts("Amir") == [0.0]


def test_goalkeeper_not_in_that_team_is_rejected():
    with pytest.raises(ConfigError):
        timeline(START, {"t": 5, "goalkeepers": {"A": "Tal"}})
