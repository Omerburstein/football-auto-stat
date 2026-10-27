import numpy as np
import pytest

from tracking.calibration import Calibration, pitch_landmarks
from tracking.camera import Camera
from tracking.config import CameraConfig, PitchConfig

PITCH = PitchConfig(40, 20)
# A made-up perspective: pitch metres -> pixels.
TRUE_H = np.array([[20.0, 5.0, 100.0], [0.0, 15.0, 50.0], [0.0, 0.01, 1.0]])


def to_pixels(points):
    points = np.c_[np.asarray(points, dtype=float), np.ones(len(points))] @ TRUE_H.T
    return points[:, :2] / points[:, 2:]


def fitted() -> Calibration:
    pitch_points = np.array(list(pitch_landmarks(PITCH).values()))
    return Calibration.fit(to_pixels(pitch_points), pitch_points)


def test_landmarks_follow_pitch_size():
    landmarks = pitch_landmarks(PITCH)
    assert landmarks["corner_bottom_right"] == (40, 20)
    assert landmarks["center_spot"] == (20, 10)


def test_homography_maps_pixels_back_to_metres():
    calibration = fitted()
    points = np.array([[3.0, 4.0], [33.0, 17.5]])
    assert np.allclose(calibration.image_to_pitch(to_pixels(points)), points, atol=1e-6)
    assert calibration.reprojection_error_m < 1e-6


@pytest.mark.parametrize("image, pitch", [
    ([[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 0], [0, 1]]),          # too few
    ([[0, 0], [1, 1], [2, 2], [3, 3]], [[0, 0], [1, 1], [2, 2], [3, 3]]),  # collinear
])
def test_fit_rejects_unusable_points(image, pitch):
    with pytest.raises(ValueError):
        Calibration.fit(image, pitch)


def test_calibration_save_load_round_trip(tmp_path):
    calibration = fitted()
    calibration.save(tmp_path / "c.yaml")
    loaded = Calibration.load(tmp_path / "c.yaml")
    assert np.allclose(loaded.homography, calibration.homography)


def camera(tmp_path, **kwargs) -> Camera:
    config = CameraConfig("cam1", tmp_path / "missing.mp4", tmp_path / "no_calibration.yaml", **kwargs)
    return Camera(config, fps=25.0)


def test_frame_time_includes_sync_offset(tmp_path):
    cam = camera(tmp_path, time_offset_s=2.0)
    assert cam.frame_time(50) == pytest.approx(4.0)
    assert cam.time_to_frame(4.0) == 50
    assert cam.time_to_frame(0.0) == 0


def test_uncalibrated_camera_gives_nan_positions(tmp_path):
    cam = Camera.from_config(CameraConfig("cam1", tmp_path / "v.mp4", tmp_path / "none.yaml"))
    assert cam.calibration is None
    assert np.isnan(cam.image_to_pitch([[10, 10]])).all()


def test_foot_point_is_bottom_centre():
    assert np.allclose(Camera.foot_points([[10, 20, 30, 80]]), [[20, 80]])


def test_missing_video_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        camera(tmp_path).read_frame(0)
