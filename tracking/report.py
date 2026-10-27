"""Review report: where the pipeline is unsure, so the user knows exactly where to add enrollment clicks."""
from __future__ import annotations

from objects.game_state import GameState
from tracking.config import format_time
from tracking.gallery import PlayerGallery
from tracking.identity import Assignment
from tracking.timeline import Timeline
from tracking.tracklet import Tracklet

MAX_LISTED_UNKNOWN = 30
ENROLL_COMMAND = "py -3.13 -m tools.enroll <match.yaml>"


def build_report(timeline: Timeline, gallery: PlayerGallery, tracklets: list[Tracklet],
                 assignments: dict[str, Assignment], state: GameState, min_unknown_s: float) -> str:
    """Markdown text: per-player coverage, outfits without enrollment, and long unknown tracklets."""
    lines = ["# Tracking review", ""]
    named = [t for t in tracklets if assignments[t.key].name is not None]
    lines.append(f"- tracklets: {len(tracklets)}, identified: {len(named)} "
                 f"({sum(t.duration for t in named):.0f} s of {sum(t.duration for t in tracklets):.0f} s tracked)")
    lines.append(f"- player rows in output: {len(state.players)}, identified: {state.players.name.notna().sum()}")
    lines += ["", "## Players", "", "| player | seen (s) | tracklets | first | last |", "|---|---|---|---|---|"]
    for name in timeline.roster:
        own = [t for t in named if assignments[t.key].name == name]
        if own:
            lines.append(f"| {name} | {sum(t.duration for t in own):.1f} | {len(own)} | "
                         f"{format_time(min(t.start for t in own))} | {format_time(max(t.end for t in own))} |")
        else:
            lines.append(f"| {name} | 0 | 0 | - | - |")

    missing = [(name, period, start) for name in timeline.roster
               for period, start in enumerate(timeline.outfit_starts(name)) if not gallery.has(name, period)]
    if missing and not gallery.keys():
        lines += ["", "## No players enrolled yet", "", f"Run `{ENROLL_COMMAND} --at 00:00` and click every player."]
    elif missing:
        lines += ["", "## Outfits without enrollment (these players can't be recognised)", ""]
        lines += [f"- {name}: outfit #{period} from {format_time(start)} -> "
                  f"`{ENROLL_COMMAND} --at {format_time(start)}`" for name, period, start in missing]

    unknown = sorted((t for t in tracklets if assignments[t.key].name is None and t.duration >= min_unknown_s),
                     key=lambda t: -t.duration)
    if unknown:
        lines += ["", f"## Longest unknown tracklets ({len(unknown)} >= {min_unknown_s:g} s)",
                  "Click these players in the enrollment tool at the given time, then re-run.", "",
                  "| camera | track | from | to | enroll at |", "|---|---|---|---|---|"]
        for t in unknown[:MAX_LISTED_UNKNOWN]:
            lines.append(f"| {t.camera_id} | {t.key} | {format_time(t.start)} | {format_time(t.end)} | "
                         f"{format_time((t.start + t.end) / 2)} |")
    return "\n".join(lines) + "\n"
