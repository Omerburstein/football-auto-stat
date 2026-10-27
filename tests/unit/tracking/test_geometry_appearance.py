import numpy as np
import pytest

from tests.helpers import GRASS_BGR
from tracking.appearance import ColorHistogramEncoder, cosine_distance, mean_embedding
from tracking.geometry import iou_matrix, isolated_mask


def test_iou_of_identical_disjoint_and_half_overlapping_boxes():
    ious = iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10], [20, 20, 30, 30], [5, 0, 15, 10]])
    assert np.allclose(ious, [[1.0, 0.0, 1 / 3]])


def test_iou_handles_empty_and_zero_area_boxes():
    assert iou_matrix(np.empty((0, 4)), [[0, 0, 1, 1]]).shape == (0, 1)
    assert iou_matrix([[0, 0, 0, 0]], [[0, 0, 0, 0]])[0, 0] == 0.0


def test_isolated_mask_flags_overlapping_boxes():
    mask = isolated_mask([[0, 0, 10, 10], [5, 0, 15, 10], [50, 50, 60, 60]], max_iou=0.05)
    assert mask.tolist() == [False, False, True]


def player_image(shirt, shorts, size=(60, 24)) -> tuple[np.ndarray, np.ndarray]:
    """Grass image with one player box; the box is wider than the player so grass is inside it."""
    image = np.full((100, 100, 3), GRASS_BGR, dtype=np.uint8)
    image[20:50, 40:64] = shirt
    image[50:80, 40:64] = shorts
    return image, np.array([[34, 20, 70, 80]], dtype=float)


ENCODER = ColorHistogramEncoder()


def embed(shirt, shorts) -> np.ndarray:
    return ENCODER.encode(*player_image(shirt, shorts))


def test_embeddings_are_normalised_with_fixed_size():
    vector = embed((0, 0, 220), (255, 255, 255))
    assert vector.shape == (1, ENCODER.dim)
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_same_outfit_is_close_different_outfits_are_far():
    red = embed((0, 0, 220), (255, 255, 255))
    red_again = embed((10, 10, 210), (245, 245, 245))   # slightly different lighting
    blue = embed((220, 40, 0), (20, 20, 20))
    assert cosine_distance(red, red_again)[0, 0] < 0.1
    assert cosine_distance(red, blue)[0, 0] > 0.8


def test_white_and_black_kits_are_told_apart():
    white = embed((255, 255, 255), (255, 255, 255))
    black = embed((15, 15, 15), (15, 15, 15))
    assert cosine_distance(white, black)[0, 0] > 0.8


def test_grass_around_the_player_is_ignored():
    image, box = player_image((0, 0, 220), (255, 255, 255))
    tight = ENCODER.encode(image, np.array([[40, 20, 64, 80]], dtype=float))
    assert cosine_distance(tight, ENCODER.encode(image, box))[0, 0] < 0.05


def test_no_boxes_gives_empty_array():
    image, _ = player_image((0, 0, 220), (0, 0, 0))
    assert ENCODER.encode(image, np.empty((0, 4))).shape == (0, ENCODER.dim)


def test_mean_embedding_is_normalised():
    assert np.linalg.norm(mean_embedding(np.array([[1.0, 0], [0, 1.0]]))) == pytest.approx(1.0)
