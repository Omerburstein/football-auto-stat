import numpy as np

from tests.helpers import OUTFITS, make_tracklet
from tracking.constraints import CrossCameraAgreement, MaxSpeed, NoOverlapSameCamera
from tracking.gallery import PlayerGallery


def test_same_camera_overlap_means_different_players():
    a, b = make_tracklet("cam1:1", 0, 20), make_tracklet("cam1:2", 10, 20)
    c = make_tracklet("cam1:3", 30, 10)
    rule = NoOverlapSameCamera()
    assert not rule.compatible(a, b)
    assert rule.compatible(a, c)
    assert rule.compatible(a, make_tracklet("cam2:1", 0, 20, camera="cam2"))


def test_max_speed_rejects_impossible_jumps():
    rule = MaxSpeed(max_speed_mps=8.0)
    first = make_tracklet("cam1:1", 0, 10, x_start=0)               # ends at frame 9 (0.9 s), x = 0
    near = make_tracklet("cam1:2", 19, 10, x_start=5)              # 1 s later, 5 m away
    far = make_tracklet("cam1:3", 19, 10, x_start=30)              # 1 s later, 30 m away
    assert rule.compatible(first, near)
    assert not rule.compatible(first, far)
    assert not rule.compatible(far, first)                          # order of arguments doesn't matter


def test_max_speed_uses_box_height_when_uncalibrated():
    first, far = make_tracklet("cam1:1", 0, 10), make_tracklet("cam1:2", 11, 10, x_start=30)
    for tracklet in (first, far):
        tracklet.pitch_xy[:] = np.nan
    # boxes are 50 px high (= 1.8 m) and the jump is 300 px (~10.8 m) in 0.2 s
    assert not MaxSpeed(8.0).compatible(first, far)


def test_cross_camera_overlap_needs_matching_positions():
    rule = CrossCameraAgreement(max_distance_m=1.5)
    cam1 = make_tracklet("cam1:1", 0, 20, x_start=10)
    same_spot = make_tracklet("cam2:1", 5, 20, camera="cam2", x_start=10.5)
    elsewhere = make_tracklet("cam2:2", 5, 20, camera="cam2", x_start=25)
    assert rule.compatible(cam1, same_spot)
    assert not rule.compatible(cam1, elsewhere)
    same_spot.pitch_xy[:] = np.nan
    assert not rule.compatible(cam1, same_spot)


def test_gallery_distance_per_outfit_period():
    gallery = PlayerGallery()
    gallery.add("Dan", 0, OUTFITS["red"])
    gallery.add("Dan", 1, OUTFITS["blue"])
    assert gallery.distance("Dan", 0, OUTFITS["red"]) == 0.0
    assert gallery.distance("Dan", 1, OUTFITS["red"]) == 1.0
    assert gallery.distance("Dan", 2, OUTFITS["red"]) == np.inf
    assert gallery.distance("Tal", 0, OUTFITS["red"]) == np.inf
    assert gallery.has("Dan", 1) and not gallery.has("Dan", 2)


def test_gallery_copy_is_independent():
    gallery = PlayerGallery()
    gallery.add("Dan", 0, OUTFITS["red"])
    clone = gallery.copy()
    clone.add("Tal", 0, OUTFITS["blue"])
    assert not gallery.has("Tal", 0)
    assert set(clone.keys()) == {("Dan", 0), ("Tal", 0)}
