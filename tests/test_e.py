"""Snapshot tests for Rigol 1000E-family `wfmconvert info` output."""

from pathlib import Path
import wave

import pytest
import RigolWFM.wfm

from tests.cli_helpers import assert_wfmconvert_info_snapshot

_E_INFO_CASES = [
    "DS1102E-A",
    "DS1102E-B",
    "DS1102E-C",
    "DS1102E-D",
    "DS1102E-E",
    "DS1102E-F",
    "DS1102E-G",
    "DS1052E",
    "DS1000E-A",
    "DS1000E-B",
    "DS1000E-C",
    "DS1000E-D",
]

_DS1102E_POINT_CASES = [
    ("DS1102E-A", (16384,)),
    ("DS1102E-B", (16384,)),
    ("DS1102E-C", (16384,)),
    ("DS1102E-D", (8192, 8192)),
    ("DS1102E-E", (1048576,)),
    ("DS1102E-F", (16384,)),
    ("DS1102E-G", (16384,)),
]


@pytest.mark.parametrize("stem", _E_INFO_CASES)
def test_wfmconvert_e_info_matches_snapshot(stem):
    """`wfmconvert E info` should match the checked-in snapshot output."""
    assert_wfmconvert_info_snapshot("E", stem, "e")


@pytest.mark.parametrize("stem, expected_points", _DS1102E_POINT_CASES)
def test_ds1102e_channel_counts_follow_memory_depth(stem, expected_points):
    """DS1102E captures should preserve the full stored analog memory depth."""
    waveform = RigolWFM.wfm.Wfm.from_file(f"tests/files/wfm/{stem}.wfm", "E")
    actual_points = tuple(channel.points for channel in waveform.channels if channel.enabled)
    assert actual_points == expected_points


@pytest.mark.parametrize("stem, _expected_points", _DS1102E_POINT_CASES)
def test_ds1102e_time_grid_matches_sample_period(stem, _expected_points):
    """Adjacent timestamps should be spaced by the reported sample period."""
    waveform = RigolWFM.wfm.Wfm.from_file(f"tests/files/wfm/{stem}.wfm", "E")
    for channel in waveform.channels:
        if channel.enabled:
            assert channel.times[1] - channel.times[0] == pytest.approx(channel.seconds_per_point)


def test_wav_export_accepts_pathlike_output(tmp_path):
    """`Wfm.wav()` should accept `pathlib.Path` destinations as documented."""
    waveform = RigolWFM.wfm.Wfm.from_file("tests/files/wfm/DS1102E-D.wfm", "E")

    wav_path = Path(tmp_path) / "ds1102e-d.wav"
    waveform.wav(wav_path, channel=1, scale="auto")

    assert wav_path.is_file()
    with wave.open(str(wav_path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getnframes() == 8192


def test_pwl_export_matches_the_channel_data():
    """`Wfm.pwl()` should tabulate one channel against a zero-based time axis."""
    waveform = RigolWFM.wfm.Wfm.from_file("tests/files/wfm/DS1102E-D.wfm", "E", selected="1")
    channel = waveform.channels[0]

    rows = waveform.pwl().splitlines()

    assert len(rows) == channel.points
    first_time, first_volts = rows[0].split("\t")
    assert float(first_time) == 0.0
    assert float(first_volts) == pytest.approx(channel.volts[0], abs=1e-6)

    last_time, last_volts = rows[-1].split("\t")
    assert float(last_time) == pytest.approx(channel.times[-1] - channel.times[0], rel=1e-6)
    assert float(last_volts) == pytest.approx(channel.volts[-1], abs=1e-6)


def test_pwl_export_ends_with_a_newline():
    """LTspice reads the table line by line, so the last row needs terminating."""
    waveform = RigolWFM.wfm.Wfm.from_file("tests/files/wfm/DS1102E-D.wfm", "E", selected="1")

    assert waveform.pwl().endswith("\n")


def test_pwl_export_rejects_more_than_one_channel():
    """A PWL source drives one node, so two selected channels are ambiguous."""
    waveform = RigolWFM.wfm.Wfm.from_file("tests/files/wfm/DS1102E-D.wfm", "E")

    with pytest.raises(ValueError, match="only one channel"):
        waveform.pwl()


def test_pwl_export_is_empty_without_channels():
    """With nothing selected there is no trace to write."""
    waveform = RigolWFM.wfm.Wfm.from_file("tests/files/wfm/DS1102E-D.wfm", "E", selected="3")

    assert waveform.pwl() == ""


def test_wav_export_rejects_channel_that_is_not_selected(tmp_path):
    """`Wfm.wav()` should reject channels excluded by the `selected=` filter."""
    waveform = RigolWFM.wfm.Wfm.from_file("tests/files/wfm/DS1102E-D.wfm", "E", selected="1")

    with pytest.raises(ValueError, match="enabled/selected"):
        waveform.wav(tmp_path / "not-selected.wav", channel=2, scale="scope")
