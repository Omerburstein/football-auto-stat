"""A small OpenCV window that shows a (scaled down) frame and reports clicks in full-frame pixels."""
from __future__ import annotations

import cv2
import numpy as np

MAX_DISPLAY_WIDTH = 1280
FONT = cv2.FONT_HERSHEY_SIMPLEX


class ClickWindow:
    """Shows images and collects left clicks, converted back to full-resolution pixels."""

    def __init__(self, title: str):
        self.title = title
        self.scale = 1.0
        self.clicks: list[tuple[float, float]] = []   # unhandled clicks, full-frame pixels
        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, self._on_mouse)

    def show(self, image: np.ndarray) -> None:
        self.scale = min(1.0, MAX_DISPLAY_WIDTH / image.shape[1])
        cv2.imshow(self.title, cv2.resize(image, None, fx=self.scale, fy=self.scale) if self.scale < 1 else image)

    def key(self, delay_ms: int = 50) -> str | None:
        """The pressed key as a character, or None."""
        code = cv2.waitKey(delay_ms) & 0xFF
        return None if code == 0xFF else chr(code)

    def pop_click(self) -> tuple[float, float] | None:
        return self.clicks.pop(0) if self.clicks else None

    def close(self) -> None:
        cv2.destroyWindow(self.title)

    def _on_mouse(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicks.append((x / self.scale, y / self.scale))


def draw_label(image: np.ndarray, text: str, origin, color) -> None:
    """Text with a dark outline so it stays readable on grass and on white shirts."""
    x, y = int(origin[0]), int(origin[1])
    cv2.putText(image, text, (x, y), FONT, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), FONT, 0.6, color, 2, cv2.LINE_AA)


def draw_help(image: np.ndarray, lines: list[str]) -> None:
    for i, line in enumerate(lines):
        draw_label(image, line, (10, 25 + 25 * i), (255, 255, 255))
