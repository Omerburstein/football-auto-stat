"""Cutting tracklets where the online tracker most likely swapped two players."""
from __future__ import annotations

from itertools import pairwise

from tracking.appearance import cosine_distance, mean_embedding
from tracking.enrollment import Enrollment
from tracking.geometry import iou_matrix
from tracking.settings import TrackingSettings
from tracking.tracklet import Tracklet


def _cut_between(frame_before: int, frame_after: int) -> int:
    """First frame of the second half when cutting between two frames (frame_before < frame_after)."""
    return (frame_before + frame_after) // 2 + 1


def split_by_appearance(tracklet: Tracklet, settings: TrackingSettings) -> list[Tracklet]:
    """Recursively split where the mean outfit before and after differ by more than `split_distance`."""
    embeddings, min_side = tracklet.embeddings, settings.min_split_samples
    if len(embeddings) < 2 * min_side:
        return [tracklet]
    best_index, best_distance = None, settings.split_distance
    for index in range(min_side, len(embeddings) - min_side + 1):
        distance = cosine_distance(mean_embedding(embeddings[:index]), mean_embedding(embeddings[index:]))[0, 0]
        if distance > best_distance:
            best_index, best_distance = index, distance
    if best_index is None:
        return [tracklet]
    samples = tracklet.sample_frames
    first, second = tracklet.split_at(_cut_between(samples[best_index - 1], samples[best_index]))
    return split_by_appearance(first, settings) + split_by_appearance(second, settings)


def anchors_of(tracklet: Tracklet, enrollment: Enrollment, min_iou: float) -> list[tuple[int, str]]:
    """(frame, name) of every enrollment click that lands on this tracklet's box, ordered by frame."""
    anchors = []
    for sample in enrollment.samples:
        if sample.camera_id != tracklet.camera_id:
            continue
        box = tracklet.box_at(sample.frame)
        if box is not None and iou_matrix(box, sample.box)[0, 0] >= min_iou:
            anchors.append((sample.frame, sample.name))
    return sorted(anchors)


def split_by_anchors(tracklet: Tracklet, anchors: list[tuple[int, str]]) -> list[tuple[Tracklet, str | None]]:
    """Split between enrollment clicks of different names on one tracklet (a swap the user saw),
    returning each piece with the name pinned to it (None for a tracklet without clicks)."""
    pieces, current = [], tracklet
    for (frame_a, name_a), (frame_b, name_b) in pairwise(anchors):
        if name_a != name_b:
            first, current = current.split_at(_cut_between(frame_a, frame_b))
            pieces.append((first, name_a))
    pieces.append((current, anchors[-1][1] if anchors else None))
    return pieces
