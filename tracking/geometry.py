"""Box helpers shared by the tracking stages. Boxes are (x1, y1, x2, y2) pixels."""
import numpy as np


def iou_matrix(boxes_a, boxes_b) -> np.ndarray:
    """(N, M) intersection-over-union of every box in `boxes_a` with every box in `boxes_b`."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.clip(bottom_right - top_left, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def isolated_mask(boxes, max_iou: float) -> np.ndarray:
    """True for boxes that overlap no other box by more than `max_iou` (clean appearance crops)."""
    ious = iou_matrix(boxes, boxes)
    np.fill_diagonal(ious, 0.0)
    return ious.max(axis=1, initial=0.0) <= max_iou
