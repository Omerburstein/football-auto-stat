"""Who plays where, as goalkeeper or not, and in which outfit - at any moment of the game.

Teams are never inferred from shirt colours: they come only from the timeline in match.yaml.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field, replace

from tracking.config import ConfigError, TimelineEvent, format_time

GOALKEEPER = "goalkeeper"
FIELD_PLAYER = "player"


@dataclass(frozen=True)
class TimelineState:
    """The line-up from `start` seconds until the next event."""
    start: float
    teams: dict[str, str] = field(default_factory=dict)        # player -> team
    goalkeepers: dict[str, str] = field(default_factory=dict)  # team -> player
    on_field: frozenset[str] = frozenset()
    outfit_periods: dict[str, int] = field(default_factory=dict)  # player -> outfit index (0 = first)


class Timeline:
    """Answers per-time questions about the roster by replaying the timeline events in order."""

    def __init__(self, events: list[TimelineEvent], roster: list[str]):
        self._roster = list(roster)
        initial = TimelineState(start=0.0, on_field=frozenset(roster))
        self._states = [initial]
        for event in sorted(events, key=lambda e: e.t):
            self._states.append(self._apply(self._states[-1], event))
        self._starts = [s.start for s in self._states]

    @property
    def roster(self) -> list[str]:
        return list(self._roster)

    @property
    def change_times(self) -> list[float]:
        """Times (s) at which something in the line-up changes."""
        return self._starts[1:]

    def state_at(self, t: float) -> TimelineState:
        return self._states[bisect.bisect_right(self._starts, t) - 1]

    def team_of(self, name: str, t: float) -> str | None:
        return self.state_at(t).teams.get(name)

    def role_of(self, name: str, t: float) -> str:
        state = self.state_at(t)
        return GOALKEEPER if state.goalkeepers.get(state.teams.get(name)) == name else FIELD_PLAYER

    def is_on_field(self, name: str, t: float) -> bool:
        return name in self.state_at(t).on_field

    def outfit_period(self, name: str, t: float) -> int:
        """Index of the outfit `name` wears at `t`; increases by one at every outfit_change event."""
        return self.state_at(t).outfit_periods.get(name, 0)

    def outfit_starts(self, name: str) -> list[float]:
        """Start time (s) of each outfit of `name`: [0.0] plus one entry per outfit_change."""
        starts, period = [0.0], 0
        for state in self._states:
            if state.outfit_periods.get(name, 0) > period:
                period = state.outfit_periods[name]
                starts.append(state.start)
        return starts

    @staticmethod
    def _apply(state: TimelineState, event: TimelineEvent) -> TimelineState:
        teams, on_field = dict(state.teams), set(state.on_field)
        if event.teams is not None:
            # A full line-up also says who is on the field.
            teams = {name: team for team, names in event.teams.items() for name in names}
            on_field = set(teams)
        teams.update(event.moves)
        on_field |= set(event.moves) | set(event.sub_in)
        on_field -= set(event.sub_out)

        # Goalkeepers who left their team or the field lose the role.
        goalkeepers = {team: gk for team, gk in state.goalkeepers.items()
                       if teams.get(gk) == team and gk in on_field}
        for team, gk in event.goalkeepers.items():
            if teams.get(gk) != team:
                raise ConfigError(f"{format_time(event.t)}: goalkeeper {gk} is not in team {team}")
            goalkeepers[team] = gk

        periods = dict(state.outfit_periods)
        for name in event.outfit_change:
            periods[name] = periods.get(name, 0) + 1
        return replace(state, start=event.t, teams=teams, goalkeepers=goalkeepers,
                       on_field=frozenset(on_field), outfit_periods=periods)
