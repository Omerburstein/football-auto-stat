"""Calibrate a fixed camera: click pitch landmarks so positions can be measured in metres.

    py -3.13 -m tools.calibrate matches/<game>/match.yaml [--camera cam1] [--at 00:10]

Left-click a point you can identify on the pitch, then type in the terminal which landmark it is:
its number from the printed list, or 'x,y' in metres for any other point (x along the length from
the left goal line, y across from the top touchline). Click >= 4 points spread over the pitch.
Keys in the window: u = undo, s = save, q = quit without saving.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from tools.window import ClickWindow, draw_help, draw_label
from tracking.calibration import MIN_POINTS, Calibration, pitch_landmarks
from tracking.camera import Camera
from tracking.config import MatchConfig, PitchConfig, parse_time

POINT_COLOR, LINE_COLOR = (0, 255, 255), (255, 0, 255)


def parse_landmark(text: str, landmarks: dict[str, tuple[float, float]]) -> tuple[float, float] | None:
    """Pitch point from a landmark number, a landmark name or 'x,y' metres; None if not understood."""
    text = text.strip()
    names = list(landmarks)
    if text.isdigit() and 1 <= int(text) <= len(names):
        return landmarks[names[int(text) - 1]]
    if text in landmarks:
        return landmarks[text]
    try:
        x, y = (float(v) for v in text.split(","))
        return x, y
    except ValueError:
        return None


def pitch_outline(pitch: PitchConfig) -> list[np.ndarray]:
    """Pitch lines in metres (outline and halfway line) for drawing the calibration result."""
    length, width = pitch.length_m, pitch.width_m
    return [np.array([[0, 0], [length, 0], [length, width], [0, width], [0, 0]], dtype=np.float64),
            np.array([[length / 2, 0], [length / 2, width]], dtype=np.float64)]


class CalibrationTool:
    """Interactive point clicking on one frame, producing a Calibration."""

    def __init__(self, image: np.ndarray, pitch: PitchConfig):
        self.image = image
        self.pitch = pitch
        self.landmarks = pitch_landmarks(pitch)
        self.image_points: list[tuple[float, float]] = []
        self.pitch_points: list[tuple[float, float]] = []

    def run(self) -> Calibration | None:
        """Interactive loop; returns the calibration when the user saves, None when they quit."""
        print("Landmarks:", *(f"  {i}. {name} {xy}" for i, (name, xy) in enumerate(self.landmarks.items(), 1)), sep="\n")
        window = ClickWindow("calibrate")
        try:
            while True:
                window.show(self._draw())
                key = window.key()
                click = window.pop_click()
                if click is not None:
                    self._ask_landmark(click)
                if key == "u" and self.image_points:
                    self.image_points.pop()
                    self.pitch_points.pop()
                elif key == "s":
                    calibration = self._fit()
                    if calibration is not None:
                        return calibration
                    print(f"Need >= {MIN_POINTS} points that are not on one line.")
                elif key == "q":
                    return None
        finally:
            window.close()

    def _ask_landmark(self, click: tuple[float, float]) -> None:
        point = parse_landmark(input(f"Point at pixel ({click[0]:.0f}, {click[1]:.0f}) is landmark #/name/'x,y': "),
                               self.landmarks)
        if point is None:
            print("Not understood - point ignored.")
            return
        self.image_points.append(click)
        self.pitch_points.append(point)
        calibration = self._fit()
        if calibration is not None:
            print(f"Reprojection error: {calibration.reprojection_error_m:.2f} m")

    def _fit(self) -> Calibration | None:
        try:
            return Calibration.fit(self.image_points, self.pitch_points)
        except ValueError:
            return None

    def _draw(self) -> np.ndarray:
        canvas = self.image.copy()
        calibration = self._fit()
        if calibration is not None:
            inverse = np.linalg.inv(calibration.homography)
            for line in pitch_outline(self.pitch):
                pixels = cv2.perspectiveTransform(line.reshape(-1, 1, 2), inverse).reshape(-1, 2)
                cv2.polylines(canvas, [pixels.astype(np.int32)], False, LINE_COLOR, 2)
        for (x, y), (px, py) in zip(self.image_points, self.pitch_points):
            cv2.circle(canvas, (int(x), int(y)), 6, POINT_COLOR, -1)
            draw_label(canvas, f"({px:g}, {py:g})", (x + 8, y - 8), POINT_COLOR)
        draw_help(canvas, [f"points: {len(self.image_points)}  click = add, u = undo, s = save, q = quit"])
        return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("match", type=Path)
    parser.add_argument("--camera", help="camera id (default: the first camera)")
    parser.add_argument("--at", default="0", help="video time of the frame to click on, e.g. 00:10")
    args = parser.parse_args()

    config = MatchConfig.load(args.match)
    camera_config = config.camera(args.camera) if args.camera else config.cameras[0]
    camera = Camera(camera_config)
    image = camera.read_frame(camera.time_to_frame(parse_time(args.at)))
    if image is None:
        raise SystemExit(f"No frame at {args.at} in {camera.video}")
    calibration = CalibrationTool(image, config.pitch).run()
    if calibration is not None:
        calibration.save(camera_config.calibration)
        print(f"Saved {camera_config.calibration} (error {calibration.reprojection_error_m:.2f} m)")


if __name__ == "__main__":
    main()
