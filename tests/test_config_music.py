"""music: config parsing and the validations that catch a bad block early.

PUBLIC MIRROR: neutral placeholders only — no real device, path, or agent name.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from claw.config import MusicConfig, _parse_music, _validate_music


def _out(**over):
    d = dict(id="room-a", name="Room A", mpv_device="coreaudio/DeviceA")
    d.update(over)
    return d


def test_absent_block_disables_without_configuring_anything():
    cfg = _parse_music(None)
    assert cfg.enabled is False
    assert cfg.outputs == ()


def test_scalars_are_coerced_to_their_declared_types():
    cfg = _parse_music({
        "enabled": True,
        "library_root": "/tmp/library",
        "mpv_socket": "/tmp/mpv.sock",
        "exposed_to": ["agent-a"],
        "connect_timeout_s": 9,
    })
    assert isinstance(cfg.library_root, Path)
    assert isinstance(cfg.mpv_socket, Path)
    assert cfg.exposed_to == ("agent-a",)
    assert cfg.connect_timeout_s == 9.0


def test_a_misspelled_scalar_key_fails_the_load():
    # The whole point of the strict splat: this block is rendered from one
    # ansible variable into two files, and a silently-dropped key is how the
    # two renders drift apart.
    with pytest.raises(TypeError):
        _parse_music({"enabled": True, "conect_timeout_s": 5})


def test_a_misspelled_output_key_fails_the_load():
    with pytest.raises(TypeError):
        _parse_music({"enabled": True, "outputs": [_out(mpv_devise="x")]})


def test_absent_bluetooth_address_means_no_connect_step():
    cfg = _parse_music({"enabled": True, "outputs": [_out()]})
    assert cfg.outputs[0].bluetooth_address is None
    assert cfg.outputs[0].default is False


def test_by_id_and_default_output():
    cfg = _parse_music({
        "enabled": True,
        "outputs": [_out(), _out(id="room-b", name="Room B", default=True)],
    })
    assert cfg.by_id("room-b").name == "Room B"
    assert cfg.by_id("nope") is None
    assert cfg.default_output.id == "room-b"


def test_no_default_output_is_allowed_at_parse_time():
    cfg = _parse_music({"enabled": True, "outputs": [_out()]})
    assert cfg.default_output is None


# --- validation ---------------------------------------------------------


def _cfg_with(make_cfg, tmp_path, music):
    return make_cfg(tmp_path, music=music)


def test_exposed_to_must_name_a_real_agent(make_cfg, tmp_path):
    music = _parse_music({"enabled": True, "exposed_to": ["ghost"], "outputs": [_out()]})
    with pytest.raises(ValueError, match="unknown agent"):
        _validate_music(_cfg_with(make_cfg, tmp_path, music))


def test_enabled_with_no_outputs_is_a_config_error(make_cfg, tmp_path):
    music = _parse_music({"enabled": True})
    with pytest.raises(ValueError, match="no outputs"):
        _validate_music(_cfg_with(make_cfg, tmp_path, music))


def test_duplicate_output_ids_are_refused(make_cfg, tmp_path):
    # Otherwise which speaker answers depends on list order.
    music = _parse_music({"enabled": True, "outputs": [_out(), _out(name="Room A again")]})
    with pytest.raises(ValueError, match="duplicate output id"):
        _validate_music(_cfg_with(make_cfg, tmp_path, music))


def test_two_defaults_are_refused(make_cfg, tmp_path):
    music = _parse_music({
        "enabled": True,
        "outputs": [_out(default=True), _out(id="room-b", name="Room B", default=True)],
    })
    with pytest.raises(ValueError, match="more than one output marked default"):
        _validate_music(_cfg_with(make_cfg, tmp_path, music))


def test_a_disabled_block_is_not_validated(make_cfg, tmp_path):
    # A deployment that turned music off should not have to keep its outputs
    # coherent to boot.
    music = MusicConfig(enabled=False, exposed_to=("ghost",))
    _validate_music(_cfg_with(make_cfg, tmp_path, music))


# --- the dj block ------------------------------------------------------

def test_dj_defaults_to_absent_and_off():
    from claw.config import MusicConfig
    assert MusicConfig().dj.enabled is False


def test_dj_rejects_a_drive_below_unity():
    from claw import config as c
    with pytest.raises(ValueError, match="drive must be >= 1.0"):
        c._validate_dj(c.DjConfig(enabled=True, drive=0.8))


def test_dj_rejects_a_positive_peak_ceiling():
    # It is headroom below full scale, not a target to reach for.
    from claw import config as c
    with pytest.raises(ValueError, match="headroom below full"):
        c._validate_dj(c.DjConfig(enabled=True, max_true_peak_dbfs=1.0))


def test_dj_block_splats_strictly():
    from claw import config as c
    with pytest.raises(TypeError):
        c._parse_dj({"drve": 2.0})


def test_dj_block_parses_paths_and_numbers():
    from claw import config as c
    dj = c._parse_dj({"enabled": True, "drive": "1.75", "render_dir": "/tmp/x",
                      "pad_ms": "300", "max_true_peak_dbfs": "-2"})
    assert dj.enabled and dj.drive == 1.75 and dj.pad_ms == 300
    assert dj.render_dir == Path("/tmp/x") and dj.max_true_peak_dbfs == -2.0


# --- the loudness block ------------------------------------------------

def test_loudness_defaults_to_unity():
    assert MusicConfig().loudness.normalize is False
    assert _parse_music({"enabled": True}).loudness.normalize is False


def test_loudness_block_parses_numbers():
    ld = _parse_music({"loudness": {
        "normalize": True, "target_lufs": "-13", "boost_ceiling_dbtp": 0,
        "assumed_lufs": -8, "furniture_below_album_lu": 8, "furniture_max_s": 120,
    }}).loudness
    assert ld.normalize is True
    assert ld.target_lufs == -13.0 and isinstance(ld.target_lufs, float)
    assert ld.boost_ceiling_dbtp == 0.0 and ld.furniture_max_s == 120.0


def test_loudness_block_splats_strictly():
    with pytest.raises(TypeError):
        _parse_music({"loudness": {"target_lusf": -13}})


@pytest.mark.parametrize("stale", ["normalize", "target_lufs"])
def test_the_old_top_level_loudness_keys_fail_the_load(stale):
    # They moved under `loudness:`. A stale render must fail rather than play
    # at a level nobody chose.
    with pytest.raises(TypeError):
        _parse_music({"enabled": True, stale: -13 if stale == "target_lufs" else True})


def _enabled_with(loudness):
    return _parse_music({"enabled": True, "outputs": [_out()], "loudness": loudness})


@pytest.mark.parametrize("loudness,message", [
    ({"boost_ceiling_dbtp": 0.5}, "boost itself would clip"),
    ({"target_lufs": 3}, "target_lufs is in LUFS"),
    ({"assumed_lufs": 0}, "assumed_lufs is in LUFS"),
    ({"furniture_max_s": 0}, "furniture_max_s must be > 0"),
    ({"furniture_below_album_lu": -8}, "furniture_below_album_lu must be > 0"),
])
def test_loudness_values_that_cannot_mean_what_they_say_are_refused(
    make_cfg, tmp_path, loudness, message
):
    with pytest.raises(ValueError, match=message):
        _validate_music(_cfg_with(make_cfg, tmp_path, _enabled_with(loudness)))


def test_loudness_is_validated_even_while_normalisation_is_off(make_cfg, tmp_path):
    # The furniture thresholds apply to every shuffle, and a bad target found
    # on the day someone switches normalisation on is the worst time.
    music = _enabled_with({"normalize": False, "boost_ceiling_dbtp": 2})
    with pytest.raises(ValueError, match="boost itself would clip"):
        _validate_music(_cfg_with(make_cfg, tmp_path, music))



# --- the candidates block ----------------------------------------------

def test_candidates_block_parses_and_splats_strictly():
    c = _parse_music({"candidates": {"fresh_hours": "48", "exclude_genres": ["Speech"],
                                     "pool_factor": 2, "max_tracks": "40"}}).candidates
    assert c.fresh_hours == 48.0 and c.exclude_genres == ("Speech",) and c.max_tracks == 40
    with pytest.raises(TypeError):
        _parse_music({"candidates": {"fresh_hour": 48}})


@pytest.mark.parametrize("candidates,message", [
    ({"pool_factor": 0.5}, "pool_factor must be >= 1"),
    ({"fresh_hours": -1}, "fresh_hours must be >= 0"),
    ({"max_tracks": 0}, "max_tracks must be >= 1"),
])
def test_candidates_values_that_cannot_work_are_refused(make_cfg, tmp_path, candidates, message):
    music = _parse_music({"enabled": True, "outputs": [_out()], "candidates": candidates})
    with pytest.raises(ValueError, match=message):
        _validate_music(_cfg_with(make_cfg, tmp_path, music))
