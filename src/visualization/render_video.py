"""Render a match's tracking result onto its video.

    py -3.13 -m visualization.render_video matches/<game>/match.yaml [--camera cam1] [--output PATH] [--max-frames N]

Runs the tracking pipeline first (the slow online pass is cached) and writes
runs/<game>_<camera>_overlay.mp4 with the input's fps and resolution.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from objects.game_state import GameState
from tracking.camera import Camera
from tracking.config import MatchConfig
from tracking.pipeline import TrackingPipeline
from visualization.overlay_renderer import OverlayRenderer

OUTPUT_DIR = Path("runs")


def render(camera: Camera, state: GameState, output: Path, max_frames: int | None = None) -> int:
    """Write the annotated video; returns the number of frames written."""
    capture = cv2.VideoCapture(str(camera.video))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), camera.fps, (width, height))
    renderer, frame = OverlayRenderer(), 0
    while max_frames is None or frame < max_frames:
        ok, image = capture.read()
        if not ok:
            break
        frame_state = state.frame(camera.id, frame)
        frame_state.time_s = float(camera.frame_time(frame))
        writer.write(renderer.render(image, frame_state))
        frame += 1
    capture.release()
    writer.release()
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("match", type=Path)
    parser.add_argument("--camera", help="camera id (default: the first camera)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    config = MatchConfig.load(args.match)
    camera_config = config.camera(args.camera) if args.camera else config.cameras[0]
    result = TrackingPipeline(config).run(args.max_frames)
    output = args.output or OUTPUT_DIR / f"{config.path.parent.name}_{camera_config.id}_overlay.mp4"
    frames = render(Camera.from_config(camera_config), result.state, output, args.max_frames)
    print(f"Wrote {frames} frames to {output}")


if __name__ == "__main__":
    main()
