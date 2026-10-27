import numpy as np
import pytest

from objects.game_state import BallState, FrameState, PlayerState
from tools.calibrate import parse_landmark, pitch_outline
from tools.enroll import parse_name
from tracking.calibration import pitch_landmarks
from tracking.config import PitchConfig
from visualization.overlay_renderer import UNKNOWN_COLOR, OverlayRenderer, color_for

LANDMARKS = pitch_landmarks(PitchConfig(40, 20))


@pytest.mark.parametrize("text, point", [("1", (0.0, 0.0)), ("7", (20.0, 10.0)), ("halfway_top", (20.0, 0.0)),
                                         (" 12.5, 3 ", (12.5, 3.0)), ("99", None), ("middle", None), ("", None)])
def test_parse_landmark(text, point):
    assert parse_landmark(text, LANDMARKS) == point


def test_pitch_outline_closes_the_rectangle():
    outline, halfway = pitch_outline(PitchConfig(40, 20))
    assert outline[0].tolist() == outline[-1].tolist()
    assert halfway[:, 0].tolist() == [20, 20]


@pytest.mark.parametrize("text, name", [("2", "Dan"), ("dan", "Dan"), ("Tal", "Tal"), ("0", None), ("Bob", None)])
def test_parse_name(text, name):
    assert parse_name(text, ["Amir", "Dan", "Tal"]) == name


def test_colors_are_stable_and_unknown_is_grey():
    assert color_for("Dan") == color_for("Dan")
    assert color_for(None) == UNKNOWN_COLOR


def test_renderer_draws_without_changing_the_input():
    image = np.zeros((200, 300, 3), np.uint8)
    state = FrameState("cam1", 3, 0.3, [PlayerState("cam1:1", "Dan", "A", "goalkeeper", np.array([50, 50, 80, 120.0]),
                                                    np.array([np.nan, np.nan]), 1.0),
                                        PlayerState("cam1:2", None, None, "player", np.array([150, 50, 180, 120.0]),
                                                    np.array([1.0, 2.0]), 0.0)],
                       BallState(np.array([100, 150, 106, 156.0]), True), [np.array([0, 0, 20, 20.0])])
    rendered = OverlayRenderer().render(image, state, ["extra line"])
    assert rendered.shape == image.shape
    assert rendered.any() and not image.any()
