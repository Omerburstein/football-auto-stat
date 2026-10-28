"""Appearance embeddings of player crops - how an outfit looks, as an L2-normalised vector.

The default encoder uses colour histograms of the shirt and shorts areas. It needs no extra model
and separates different outfits well in pickup games, where every player dresses differently.
A learned ReID encoder (e.g. OSNet from BoxMOT) can be added by implementing AppearanceEncoder.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

# Crop regions as fractions of the box height: shirt, then shorts/socks.
BODY_PARTS = ((0.15, 0.50), (0.50, 0.85))
CENTER_WIDTH = (0.2, 0.8)       # drop the box sides, which are mostly background
HUE_BINS = 12
CHROMATIC_SAT = 50              # HSV saturation below this counts as white/grey/black
VALUE_BINS = 4                  # brightness bins for those achromatic pixels
GRASS_HUE = (35, 85)            # OpenCV hue range (0-180) of green grass
GRASS_MIN_SAT = 60


class AppearanceEncoder(ABC):
    """Turns player boxes of one image into comparable appearance vectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier stored with cached embeddings (cache is invalid when the encoder changes)."""

    @abstractmethod
    def encode(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """(N, 4) boxes -> (N, D) L2-normalised embeddings."""


class ColorHistogramEncoder(AppearanceEncoder):
    """Hue x saturation histogram for coloured pixels plus a brightness histogram for white/black,
    per body part, with grass pixels masked out. Square-rooted so cosine similarity behaves like
    the Hellinger (Bhattacharyya) similarity of the histograms."""

    name = "color_hist_v1"

    def encode(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return np.array([self._encode_box(hsv, box) for box in np.asarray(boxes).reshape(-1, 4)]
                        ).reshape(-1, self.dim)

    @property
    def dim(self) -> int:
        return len(BODY_PARTS) * (HUE_BINS * 2 + VALUE_BINS)

    def _encode_box(self, hsv: np.ndarray, box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = box
        width, height = x2 - x1, y2 - y1
        left, right = int(x1 + CENTER_WIDTH[0] * width), int(x1 + CENTER_WIDTH[1] * width)
        parts = []
        for top, bottom in BODY_PARTS:
            crop = hsv[max(int(y1 + top * height), 0):int(y1 + bottom * height), max(left, 0):right]
            parts.append(self._histogram(crop.reshape(-1, 3)))
        vector = np.sqrt(np.concatenate(parts))
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    @staticmethod
    def _histogram(pixels: np.ndarray) -> np.ndarray:
        """Normalised histogram of one body part; all zeros when nothing but grass is visible."""
        hue, sat, val = (pixels[:, i].astype(np.int32) for i in range(3))
        grass = (hue >= GRASS_HUE[0]) & (hue <= GRASS_HUE[1]) & (sat >= GRASS_MIN_SAT)
        chromatic = ~grass & (sat >= CHROMATIC_SAT)
        achromatic = ~grass & (sat < CHROMATIC_SAT)
        hue_bin = np.minimum(hue[chromatic] * HUE_BINS // 180, HUE_BINS - 1)
        sat_bin = (sat[chromatic] >= 150).astype(np.int32)  # pale vs vivid colour
        colour = np.bincount(hue_bin * 2 + sat_bin, minlength=HUE_BINS * 2)
        brightness = np.bincount(np.minimum(val[achromatic] * VALUE_BINS // 256, VALUE_BINS - 1),
                                 minlength=VALUE_BINS)
        histogram = np.concatenate([colour, brightness]).astype(np.float64)
        total = histogram.sum()
        return histogram / total if total > 0 else histogram


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, D) x (M, D) normalised vectors -> (N, M) distances in [0, 2] (0 = same direction)."""
    return 1.0 - np.asarray(a).reshape(-1, np.shape(b)[-1]) @ np.asarray(b).reshape(-1, np.shape(b)[-1]).T


def mean_embedding(embeddings: np.ndarray) -> np.ndarray:
    """Normalised mean of several embeddings (a prototype of one outfit)."""
    mean = np.asarray(embeddings).mean(axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean
