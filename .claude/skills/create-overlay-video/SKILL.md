---
name: create-overlay-video
description: Run the football-auto-stat detection, tracking and stats pipeline on a video and render an output video with bounding boxes for players, ball and goals plus the relevant stats overlaid. Use when asked to create/render an overlay or demo video, or to visually check the algorithm. Optional argument - path to the input video.
argument-hint: "[input-video]"
---

# Create overlay video

Input video: `$ARGUMENTS`. If none was given, use the first file in `tests_videos/` (ask the user if there is none).

Use `py -3.13` (Python 3.13 has ultralytics and cv2).

## 1. Make sure the renderer exists

The overlay code lives in `visualization/` and must stay separate from detection and stats:

- `visualization/overlay_renderer.py` – class `OverlayRenderer` that takes a frame plus the per-frame state (players, ball, goals, stats) and returns the annotated frame. No model calls inside.
- `visualization/render_video.py` – CLI entry point: `py -3.13 -m visualization.render_video <input> [--output PATH] [--weights PATH]`. It runs the same pipeline as `main.py` (reuse its classes, do not copy logic), collects per-frame state, then writes the video.

If these do not exist yet or are missing something below, create or extend them (keep files short and OOP, per the clean-up-code skill).

## 2. What the video must show

- **Players**: box per player with its track id (and team / name when that exists). Stable colour per id (e.g. hash of the id into a palette). Highlight the player in possession (thicker box or a marker above the head).
- **Ball**: distinct colour, small filled marker at its centre; draw interpolated positions in a lighter shade so detected and filled-in frames can be told apart.
- **Goals**: box for each goal/goalpost detection.
- **Stats panel** (semi-transparent box in a corner, updated per frame): possession % per player/team, shots, goals, and any other stat the tracker currently computes. Flash a short label ("SHOT", "GOAL") on the frame where an event is detected.
- Frame number and timestamp in a corner.

## 3. Rules

- Read class ids from the model (`model.names`), not hardcoded numbers – map names `player`, `ball`, `goalpost` to the right objects.
- Keep the input FPS and resolution; write with `cv2.VideoWriter` (`mp4v`).
- Save to `outputs/<input-name>_overlay.mp4` (create `outputs/`; make sure `outputs/` is in `.gitignore`). Never overwrite the input.
- For a long video, first render the first ~10 seconds to check the result, then the full video.

## 4. Check and report

- Open a few frames of the result (extract with cv2 to `outputs/frames/` and look at them with the Read tool) and check boxes line up with players, ids are stable, and the stats panel is readable.
- Report the output path, video length, number of tracked players, final stats, and anything that looked wrong (id switches, missed ball, false detections).
