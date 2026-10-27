import numpy as np

from tests.helpers import make_tracklet
from tracking.enrollment import Enrollment, EnrollmentSample, box_containing
from tracking.online_tracker import FrameTracks
from tracking.tracklet import TrackletBuilder


class FakeCamera:
    id = "cam1"

    def frame_time(self, frames):
        return np.asarray(frames) / 10.0

    def image_to_pitch(self, points):
        return np.asarray(points, dtype=float) / 100.0


class ConstantEncoder:
    name = "constant"

    def encode(self, image, boxes):
        return np.tile([1.0, 0.0], (len(boxes), 1))


def test_box_at_and_split_at():
    tracklet = make_tracklet("cam1:1", 10, 20, embedding=[1.0, 0.0])
    assert tracklet.box_at(9) is None
    assert np.allclose(tracklet.box_at(12), tracklet.boxes[2])
    first, second = tracklet.split_at(20)
    assert (first.key, second.key) == ("cam1:1a", "cam1:1b")
    assert first.frames[-1] == 19 and second.frames[0] == 20
    assert len(first.embeddings) + len(second.embeddings) == len(tracklet.embeddings)
    assert all(first.sample_frames < 20) and all(second.sample_frames >= 20)


def test_prototype_is_none_without_samples():
    assert make_tracklet("cam1:1", 0, 5).prototype is None


def frame(idx, ids, boxes):
    return FrameTracks(idx, np.zeros((100, 100, 3), np.uint8), np.array(ids), np.array(boxes, float), None,
                       np.empty((0, 4)))


def test_builder_drops_short_tracks_and_samples_only_clean_boxes(settings):
    builder = TrackletBuilder(FakeCamera(), ConstantEncoder(), settings)
    clean, overlapping_a, overlapping_b = [0, 0, 10, 40], [50, 0, 60, 40], [52, 0, 62, 40]
    for idx in range(10):
        builder.add(frame(idx, [1, 2, 3], [clean, overlapping_a, overlapping_b]))
    builder.add(frame(10, [4], [clean]))   # a 1-frame track
    tracklets = {t.key: t for t in builder.build()}
    assert set(tracklets) == {"cam1:1", "cam1:2", "cam1:3"}
    assert len(tracklets["cam1:1"].embeddings) == 2       # frames 0 and 5
    assert len(tracklets["cam1:2"].embeddings) == 0       # always overlapping its neighbour
    assert np.allclose(tracklets["cam1:1"].times[:2], [0.0, 0.1])
    assert np.allclose(tracklets["cam1:1"].pitch_xy[0], [0.05, 0.40])   # foot point / 100


def test_enrollment_round_trip_and_replace(tmp_path):
    enrollment = Enrollment()
    enrollment.add(EnrollmentSample("Dan", "cam1", 5, np.array([1, 2, 3, 4.0])))
    enrollment.add(EnrollmentSample("Dan", "cam1", 5, np.array([5, 6, 7, 8.0])))   # same frame -> replaced
    enrollment.add(EnrollmentSample("Tal", "cam1", 5, np.array([1, 1, 2, 2.0])))
    enrollment.save(tmp_path / "e.yaml", fps=25)
    loaded = Enrollment.load(tmp_path / "e.yaml")
    assert loaded.names() == {"Dan", "Tal"}
    assert len(loaded.at_frame("cam1", 5)) == 2
    dan = next(s for s in loaded.samples if s.name == "Dan")
    assert dan.box.tolist() == [5, 6, 7, 8]
    loaded.remove(dan)
    assert loaded.names() == {"Tal"}


def test_missing_enrollment_file_is_empty(tmp_path):
    assert Enrollment.load(tmp_path / "nope.yaml").samples == []


def test_click_selects_smallest_box_containing_it():
    boxes = np.array([[0, 0, 100, 100], [40, 40, 60, 60]])
    assert box_containing(boxes, 50, 50) == 1
    assert box_containing(boxes, 10, 10) == 0
    assert box_containing(boxes, 200, 200) is None
    assert box_containing(np.empty((0, 4)), 1, 1) is None
