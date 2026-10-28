---
name: create-tests
description: Add unit, integration and algorithm tests (pytest) for new or changed code in the football-auto-stat project. Use after adding or changing any Python code, or when asked to write tests. Optional argument - a file or folder to cover; defaults to the code changed in git.
argument-hint: "[file-or-folder]"
---

# Create tests

Write tests for: `$ARGUMENTS`. If no target was given, cover the Python files that changed:
`git status --porcelain -- "*.py"` plus `git diff --name-only HEAD -- "*.py"` (skip files under `tests/`).

## Environment

- Use the interpreter that has the project deps: `py -3.13` (Python 3.13 has ultralytics, cv2, pytest; the default `python` 3.14 does not).
- Run tests with `py -3.13 -m pytest`.
- If `requirements-dev.txt` does not exist, create it with `pytest` in it.
- If `pytest.ini` does not exist, create it with:
  ```ini
  [pytest]
  testpaths = tests
  pythonpath = .
  markers =
      slow: needs model weights or a real video (deselect with -m "not slow")
  ```

## Layout

```
tests/
  conftest.py          # shared fixtures: synthetic boxes, fake YOLO results, small players/ball tracks
  unit/                # one test_<module>.py per source module
  integration/         # pipeline pieces working together
  algo/                # behaviour of the stats/tracking algorithms on scenarios with known answers
```

Mirror the source path in the test name: `src/objects/player.py` -> `tests/unit/objects/test_player.py`.

## What each kind of test covers

**Unit** – one class or function in isolation, no model, no video.
- Every public method of the changed code: normal case, edge cases (empty position `[]`, first frame `time_index == 0`, missing detections, a single player), and invalid input.
- Geometry helpers with hand-computed expected values (box centres, distances, speeds).

**Integration** – several modules together without the real network.
- Build fake detection results that look like ultralytics `Results` (`boxes.id`, `boxes.xyxy`, `boxes.cls` as numpy arrays) in `conftest.py`, and feed them through the same code path `src/main.py` uses (tracker creation, adding players, positions, interpolation, stats).
- Players appearing mid-video, disappearing, ball missing for some frames, frames with no ids.
- One `@pytest.mark.slow` smoke test that runs the real model (`models/weights/best_both.pt`) on a few frames of `data/videos/` and checks the output shape, skipped with `pytest.importorskip` / `skipif` when weights or video are missing.

**Algo** – the algorithm gives the right answer on a scenario whose ground truth is known.
- Build synthetic tracks frame by frame: e.g. the ball sits next to player A for 30 frames then next to player B for 10 -> possession is 75% / 25%; the ball jumps faster than `MIN_BALL_SPEED_FOR_SHOOT` next to a player -> exactly one shot for that player; ball far from everyone -> nobody gets possession.
- Check robustness: small jitter in the boxes must not change the result, a one-frame detection dropout must not split a possession.
- Use thresholds from `consts.py` instead of copying numbers, so tests follow tuning changes.

## Rules

- Tests must be deterministic: no network, fixed seeds, no dependency on test order.
- Test behaviour, not implementation details – avoid asserting on private attributes unless no public way exists.
- Keep each test short with a name that states the expected behaviour: `test_closest_player_ignores_players_beyond_min_distance`.
- Do not change source code to make a test pass. If a test exposes a real bug, keep the test, mark it `@pytest.mark.xfail(strict=True, reason="BUG: <what is wrong>")`, and report the bug to the user at the end.
- Run the full suite (`py -3.13 -m pytest -m "not slow"`, then the slow tests if weights exist) and report: tests added per kind, pass/fail counts, and any bugs found.
