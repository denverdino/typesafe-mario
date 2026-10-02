import copy

import pytest

from typesafe_mario.state import MarioStateParser


def observe(parser, positions, *, kind=6, engine_state=0, area=1, ram_present=True):
    ram = [0] * 2048
    for slot, (x, y) in enumerate(positions):
        ram[0xF + slot] = 1
        ram[0x16 + slot] = kind
        ram[0x1E + slot] = engine_state
        ram[0x6E + slot], ram[0x87 + slot] = divmod(x, 256)
        ram[0xCF + slot] = y
    return parser.parse(
        {"x_pos": 100, "y_pos": 79, "y_pixel": 176, "area": area},
        ram if ram_present else None,
    )


def test_all_five_tracks_keep_fractional_motion_including_rear_and_overhead():
    p = MarioStateParser()
    for offset in [0, 1, 2, 2, 3, 4, 4, 5, 5]:
        s = observe(p, [(50 - offset, 80)] + [(200 + i * 24 - offset, 184) for i in range(4)])
    tracks = s.to_state().get("enemy_tracks", [])
    assert len(tracks) == 5
    assert len({t["id"] for t in tracks}) == 5
    assert tracks[0]["position"] == {"x": 45, "y": 80}
    assert all(t["velocity"]["x"] == pytest.approx(-0.625) for t in tracks)
    assert all(t["observed"] for t in tracks)


def test_first_observation_unknown_and_inactive_slot_gets_new_identity():
    p = MarioStateParser()
    first = observe(p, [(200, 184)]).to_state()["enemy_tracks"][0]
    assert first["velocity"]["x"] is None
    assert observe(p, []).to_state()["enemy_tracks"] == []
    new = observe(p, [(199, 184)]).to_state()["enemy_tracks"][0]
    assert new["id"] != first["id"]
    assert new["velocity"]["x"] is None


def test_missing_telemetry_expires_and_is_not_confident_stationary():
    p = MarioStateParser()
    observe(p, [(200, 184)])
    observe(p, [(199, 184)])
    s = observe(p, [], ram_present=False)
    t = s.to_state()["enemy_tracks"][0]
    assert t["observed"] is False
    assert t["confidence"] == "low"
    for _ in range(8):
        s = observe(p, [], ram_present=False)
    assert s.to_state()["enemy_tracks"] == []


def test_kind_area_reset_and_teleport_break_identity():
    for kwargs in [{"kind": 0}, {"area": 2}, {}]:
        p = MarioStateParser()
        old = observe(p, [(200, 184)]).to_state()["enemy_tracks"][0]
        s = observe(p, [(500 if not kwargs else 199, 184)], **kwargs)
        new = s.to_state()["enemy_tracks"][0]
        assert new["id"] != old["id"]
        assert new["velocity"]["x"] is None
    old = new["id"]
    p.reset()
    assert observe(p, [(500, 184)]).to_state()["enemy_tracks"][0]["id"] != old


def test_behavior_change_keeps_identity_but_resets_motion():
    p = MarioStateParser()
    observe(p, [(200, 184)], kind=0)
    s = observe(p, [(199, 184)], kind=0)
    old = s.to_state()["enemy_tracks"][0]
    new = observe(p, [(198, 184)], kind=0, engine_state=4).to_state()["enemy_tracks"][0]
    assert new["id"] == old["id"]
    assert new["velocity"]["x"] is None


def test_reversal_discards_old_direction_and_snapshots_do_not_mutate():
    p = MarioStateParser()
    for x in [200, 199, 198, 197]:
        s = observe(p, [(x, 184)])
    old = copy.deepcopy(s.to_state()["enemy_tracks"])
    new = observe(p, [(198, 184)]).to_state()["enemy_tracks"][0]
    assert new["velocity"]["x"] > 0
    assert s.to_state()["enemy_tracks"] == old
