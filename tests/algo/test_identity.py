"""Identity assignment on synthetic tracklets whose true owner is known."""
import numpy as np
import pytest

from tests.helpers import OUTFITS, make_tracklet
from tracking.config import TimelineEvent
from tracking.enrollment import Enrollment, EnrollmentSample
from tracking.gallery import PlayerGallery
from tracking.identity import IdentityAssigner
from tracking.splitting import split_by_anchors, split_by_appearance
from tracking.timeline import Timeline

ROSTER = ["Red", "Blue", "White"]


def gallery_of(**outfits) -> PlayerGallery:
    gallery = PlayerGallery()
    for name, outfit in outfits.items():
        gallery.add(name, 0, OUTFITS[outfit])
    return gallery


def assign(tracklets, gallery, settings, events=(), enrollment=None, roster=ROSTER):
    timeline = Timeline([TimelineEvent.from_dict(e) for e in events], roster)
    pieces, assignments = IdentityAssigner(timeline, settings).assign(tracklets, gallery, enrollment or Enrollment())
    return {piece.key: assignments[piece.key].name for piece in pieces}


def test_broken_tracks_are_rejoined_by_outfit(settings):
    tracklets = [make_tracklet("cam1:1", 0, 30, OUTFITS["red"], x_start=0),
                 make_tracklet("cam1:2", 0, 30, OUTFITS["blue"], x_start=20),
                 make_tracklet("cam1:3", 35, 30, OUTFITS["red"], x_start=1, noise=0.1),
                 make_tracklet("cam1:4", 35, 30, OUTFITS["blue"], x_start=21, noise=0.1)]
    names = assign(tracklets, gallery_of(Red="red", Blue="blue"), settings)
    assert names == {"cam1:1": "Red", "cam1:2": "Blue", "cam1:3": "Red", "cam1:4": "Blue"}


def test_unknown_outfit_stays_unknown(settings):
    names = assign([make_tracklet("cam1:1", 0, 30, OUTFITS["yellow"])], gallery_of(Red="red"), settings)
    assert names == {"cam1:1": None}


def test_one_player_is_never_in_two_places_at_once(settings):
    # Two red-looking tracklets at the same time: only one of them can be Red.
    tracklets = [make_tracklet("cam1:1", 0, 30, OUTFITS["red"], x_start=0),
                 make_tracklet("cam1:2", 0, 30, OUTFITS["red"], x_start=20, noise=0.2)]
    names = assign(tracklets, gallery_of(Red="red", Blue="blue"), settings)
    assert names["cam1:1"] == "Red"          # the exact match wins
    assert names["cam1:2"] is None


def test_speed_limit_blocks_a_lookalike_far_away(settings):
    # Red leaves at x=0; 0.6 s later a red-looking track appears 30 m away -> not Red.
    # (The longer tracklet is solved first, so it is the one that gets the name.)
    tracklets = [make_tracklet("cam1:1", 0, 30, OUTFITS["red"], x_start=0),
                 make_tracklet("cam1:2", 35, 20, OUTFITS["red"], x_start=30)]
    names = assign(tracklets, gallery_of(Red="red"), settings)
    assert names["cam1:1"] == "Red" and names["cam1:2"] is None


def test_outfit_change_uses_the_gallery_of_the_new_outfit(settings):
    gallery = gallery_of(Red="red", Blue="blue")
    gallery.add("Red", 1, OUTFITS["white"])          # Red changes into white at t=10 s
    tracklets = [make_tracklet("cam1:1", 0, 50, OUTFITS["red"]),
                 make_tracklet("cam1:2", 150, 50, OUTFITS["white"], x_start=2),
                 make_tracklet("cam1:3", 150, 50, OUTFITS["red"], x_start=4)]
    names = assign(tracklets, gallery, settings, events=[{"t": 10, "outfit_change": ["Red"]}])
    assert names == {"cam1:1": "Red", "cam1:2": "Red", "cam1:3": None}


def test_players_off_the_field_are_not_assigned(settings):
    events = [{"t": 0, "teams": {"A": ["Red"], "B": ["White"]}}]
    names = assign([make_tracklet("cam1:1", 0, 30, OUTFITS["blue"])], gallery_of(Blue="blue"), settings, events)
    assert names == {"cam1:1": None}


def test_id_swap_inside_a_tracklet_is_split_and_both_halves_named(settings):
    swapped = make_tracklet("cam1:1", 0, 60, OUTFITS["red"])
    swapped.embeddings[len(swapped.embeddings) // 2:] = OUTFITS["blue"]
    names = assign([swapped], gallery_of(Red="red", Blue="blue"), settings)
    assert sorted(names.items()) == [("cam1:1a", "Red"), ("cam1:1b", "Blue")]


def test_appearance_split_keeps_consistent_tracklets_whole(settings):
    steady = make_tracklet("cam1:1", 0, 100, OUTFITS["red"], noise=0.1)
    assert len(split_by_appearance(steady, settings)) == 1


def test_enrollment_click_pins_a_tracklet_even_without_gallery_match(settings):
    tracklet = make_tracklet("cam1:1", 0, 30, OUTFITS["yellow"])
    enrollment = Enrollment([EnrollmentSample("White", "cam1", 10, tracklet.box_at(10))])
    names = assign([tracklet], gallery_of(Red="red"), settings, enrollment=enrollment)
    assert names == {"cam1:1": "White"}


def test_clicks_of_two_names_on_one_tracklet_split_it():
    tracklet = make_tracklet("cam1:1", 0, 40)
    pieces = split_by_anchors(tracklet, [(5, "Red"), (10, "Red"), (30, "Blue")])
    assert [(p.key, name) for p, name in pieces] == [("cam1:1a", "Red"), ("cam1:1b", "Blue")]
    # cut half-way between the last Red click (10) and the first Blue click (30)
    assert (pieces[0][0].frames[-1], pieces[1][0].frames[0]) == (20, 21)


def test_refinement_recognises_a_player_after_a_drift_in_appearance(settings):
    # Enrolled outfit is pure red; later views drift towards orange (lighting). Intermediate,
    # confidently matched views are added to the gallery so the drifted view is recognised too.
    reddish_orange = OUTFITS["red"] * 0.9 + OUTFITS["orange"] * 0.45
    orange_red = OUTFITS["red"] * 0.6 + OUTFITS["orange"] * 0.8
    tracklets = [make_tracklet("cam1:1", 0, 30, reddish_orange / np.linalg.norm(reddish_orange)),
                 make_tracklet("cam1:2", 40, 30, orange_red / np.linalg.norm(orange_red), x_start=1)]
    tight = settings.__class__(max_assign_distance=0.3, confident_distance=0.15)
    names = assign(tracklets, gallery_of(Red="red"), tight)
    assert names == {"cam1:1": "Red", "cam1:2": "Red"}


@pytest.mark.parametrize("seed", range(5))
def test_appearance_noise_does_not_change_who_is_who(settings, seed):
    tracklets = [make_tracklet(f"cam1:{i}", 40 * (i // 3), 30, OUTFITS[outfit], x_start=10 * (i % 3),
                               noise=0.15, seed=seed * 10 + i)
                 for i, outfit in enumerate(["red", "blue", "white"] * 3)]
    names = assign(tracklets, gallery_of(Red="red", Blue="blue", White="white"), settings)
    assert [names[f"cam1:{i}"] for i in range(9)] == ["Red", "Blue", "White"] * 3


@pytest.mark.parametrize("rounds", [0, 2])
def test_result_does_not_depend_on_refinement_when_everything_is_clear(settings, rounds):
    settings.refinement_rounds = rounds
    tracklets = [make_tracklet("cam1:1", 0, 30, OUTFITS["red"]), make_tracklet("cam1:2", 0, 30, OUTFITS["blue"], x_start=9)]
    assert assign(tracklets, gallery_of(Red="red", Blue="blue"), settings) == {"cam1:1": "Red", "cam1:2": "Blue"}
