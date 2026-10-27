"""Draws one frame's state: named player boxes, ball, goals and a small info panel."""
from __future__ import annotations

import zlib

import cv2
import numpy as np

from objects.game_state import FrameState
from tracking.config import format_time
from tracking.timeline import GOALKEEPER

PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
           (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180)]
UNKNOWN_COLOR = (150, 150, 150)
BALL_COLOR, BALL_FILLED_COLOR, GOAL_COLOR = (0, 255, 255), (170, 255, 255), (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
PANEL_ALPHA = 0.55


def color_for(name: str | None) -> tuple[int, int, int]:
    """Stable BGR colour per player name (crc32, so it is the same in every run)."""
    if name is None:
        return UNKNOWN_COLOR
    return PALETTE[zlib.crc32(name.encode()) % len(PALETTE)]


class OverlayRenderer:
    """Draws players, ball, goals and an info panel on a frame copy."""

    def render(self, image: np.ndarray, state: FrameState, panel_lines: list[str] | None = None) -> np.ndarray:
        """Annotated copy of `image`; `panel_lines` are extra lines for the info panel."""
        canvas = image.copy()
        for goal in state.goals:
            x1, y1, x2, y2 = np.asarray(goal).astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), GOAL_COLOR, 2)
        for player in state.players:
            self._draw_player(canvas, player)
        if state.ball is not None:
            x1, y1, x2, y2 = state.ball.box
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.circle(canvas, center, 7, BALL_FILLED_COLOR if state.ball.interpolated else BALL_COLOR, -1)
            cv2.circle(canvas, center, 8, (0, 0, 0), 1)
        identified = sum(p.name is not None for p in state.players)
        header = [f"{format_time(state.time_s) if not np.isnan(state.time_s) else '--:--'}  frame {state.frame}",
                  f"players {len(state.players)}  identified {identified}"]
        self._draw_panel(canvas, header + list(panel_lines or []))
        return canvas

    @staticmethod
    def _draw_player(canvas: np.ndarray, player) -> None:
        color = color_for(player.name)
        x1, y1, x2, y2 = player.box.astype(int)
        thickness = 3 if player.role == GOALKEEPER else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = player.name or "?"
        if player.team:
            label += f" ({player.team}{' GK' if player.role == GOALKEEPER else ''})"
        cv2.putText(canvas, label, (x1, y1 - 5), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (x1, y1 - 5), FONT, 0.5, color, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_panel(canvas: np.ndarray, lines: list[str]) -> None:
        width = 20 + max(cv2.getTextSize(line, FONT, 0.6, 2)[0][0] for line in lines)
        height = 15 + 25 * len(lines)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, PANEL_ALPHA, canvas, 1 - PANEL_ALPHA, 0, dst=canvas)
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (10, 28 + 25 * i), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
