"""Enroll players: click a player's box and type their name, so the system learns each outfit.

    py -3.13 -m tools.enroll matches/<game>/match.yaml [--camera cam1] [--at 00:00]

Click 3-5 views of every player (different moments / poses help), and enroll a player again after
an outfit_change in the timeline. The review report (review.md) lists where more clicks are needed.
Keys: . / , = 1 s forward / back, > / < = 10 s, d = delete mode (next click removes a sample),
s = save, q = save and quit, x = quit without saving.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from tools.window import ClickWindow, draw_help, draw_label
from tracking.camera import Camera
from tracking.config import MatchConfig, format_time, parse_time
from tracking.detection import Detector, YoloDetector
from tracking.enrollment import Enrollment, EnrollmentSample, box_containing

DETECTION_COLOR, ENROLLED_COLOR = (200, 200, 200), (0, 255, 0)
STEPS_S = {".": 1, ",": -1, ">": 10, "<": -10}


def parse_name(text: str, roster: list[str]) -> str | None:
    """Roster name from its number (1-based) or its exact / case-insensitive name; None otherwise."""
    text = text.strip()
    if text.isdigit() and 1 <= int(text) <= len(roster):
        return roster[int(text) - 1]
    matches = [name for name in roster if name.lower() == text.lower()]
    return matches[0] if matches else None


class EnrollTool:
    """Interactive naming of player boxes, frame by frame, stored in an Enrollment."""

    def __init__(self, camera: Camera, detector: Detector, enrollment: Enrollment, roster: list[str],
                 player_label: str):
        self.camera, self.detector, self.enrollment = camera, detector, enrollment
        self.roster, self.player_label = roster, player_label
        self.delete_mode = False

    def run(self, frame: int, enrollment_path: Path) -> None:
        """Interactive loop starting at `frame`; saves to `enrollment_path` on s / q."""
        window = ClickWindow("enroll")
        image, boxes = self._load(frame)
        try:
            while True:
                window.show(self._draw(image, boxes, frame))
                key, click = window.key(), window.pop_click()
                if click is not None:
                    self._handle_click(click, boxes, frame)
                if key in STEPS_S:
                    frame = max(frame + int(STEPS_S[key] * self.camera.fps), 0)
                    image, boxes = self._load(frame)
                elif key == "d":
                    self.delete_mode = not self.delete_mode
                elif key in ("s", "q"):
                    self.enrollment.save(enrollment_path, self.camera.fps)
                    print(f"Saved {len(self.enrollment.samples)} samples to {enrollment_path}")
                if key in ("q", "x"):
                    return
        finally:
            window.close()

    def _load(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        image = self.camera.read_frame(frame)
        if image is None:
            raise SystemExit(f"Frame {frame} is past the end of {self.camera.video}")
        return image, self.detector.detect(image).of_label(self.player_label).boxes

    def _handle_click(self, click, boxes: np.ndarray, frame: int) -> None:
        samples = self.enrollment.at_frame(self.camera.id, frame)
        if self.delete_mode:
            index = box_containing(np.array([s.box for s in samples]).reshape(-1, 4), *click)
            if index is not None:
                self.enrollment.remove(samples[index])
                print(f"Removed {samples[index].name}")
            self.delete_mode = False
            return
        index = box_containing(boxes, *click)
        if index is None:
            print("No player box there.")
            return
        print(*(f"  {i}. {name}" for i, name in enumerate(self.roster, 1)), sep="\n")
        name = parse_name(input("Who is this (number or name, empty = cancel)? "), self.roster)
        if name is not None:
            self.enrollment.add(EnrollmentSample(name, self.camera.id, frame, boxes[index]))
            print(f"{name} enrolled at {format_time(float(self.camera.frame_time(frame)))}")

    def _draw(self, image: np.ndarray, boxes: np.ndarray, frame: int) -> np.ndarray:
        canvas = image.copy()
        for x1, y1, x2, y2 in boxes.astype(int):
            cv2.rectangle(canvas, (x1, y1), (x2, y2), DETECTION_COLOR, 1)
        for sample in self.enrollment.at_frame(self.camera.id, frame):
            x1, y1, x2, y2 = sample.box.astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), ENROLLED_COLOR, 2)
            draw_label(canvas, sample.name, (x1, y1 - 6), ENROLLED_COLOR)
        counts = {name: sum(s.name == name for s in self.enrollment.samples) for name in self.roster}
        missing = [name for name, count in counts.items() if count == 0]
        draw_help(canvas, [
            f"{format_time(float(self.camera.frame_time(frame)))}  frame {frame}"
            + ("   DELETE MODE: click a green box" if self.delete_mode else ""),
            "click box = name it   ./, = +-1s   >/< = +-10s   d = delete   s = save   q = save+quit   x = quit",
            f"not enrolled yet: {', '.join(missing) or '-'}",
        ])
        return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("match", type=Path)
    parser.add_argument("--camera", help="camera id (default: the first camera)")
    parser.add_argument("--at", default="0", help="video time to start at, e.g. 20:00")
    args = parser.parse_args()

    config = MatchConfig.load(args.match)
    camera = Camera(config.camera(args.camera) if args.camera else config.cameras[0])
    tool = EnrollTool(camera, YoloDetector(config.detector), Enrollment.load(config.enrollment),
                      config.roster, config.detector.player_label)
    tool.run(camera.time_to_frame(parse_time(args.at)), config.enrollment)


if __name__ == "__main__":
    main()
