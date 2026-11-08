from datetime import datetime

import numpy as np
import pytest

from hplcdad import DADMeta, DADRun, DADSequence

# The fields a Chromeleon text export carries, and so should survive a round trip.
EXPORTED_FIELDS = (
    "sample_name", "sequence", "injection_datetime", "injection_number",
    "injection_volume_ul", "vial_position", "method", "injection_id",
)


def make_run(name="Brew 1", injection_id="556.smp", number=1, when=datetime(2026, 10, 21, 13, 5, 7)):
    time = 0.0002 + np.arange(300) * 0.4 / 60
    wl = np.arange(190.0, 642.0, 2.0)
    intensity = np.random.default_rng(number).normal(0, 300, (len(time), len(wl)))
    intensity[0, 0] = -1234.5678901   # thousands separator, negative
    intensity[1, 1] = 2500.25
    meta = DADMeta(
        sample_name=name,
        sequence="20261021 - Spike experiment",
        injection_datetime=when,
        injection_number=number,
        injection_volume_ul=10.0,
        vial_position="P1:F2",
        method="20261019 - Beer bitter acids (JAA)",
        injection_id=injection_id,
    )
    return DADRun(time, wl, intensity, meta)


@pytest.mark.parametrize(
    "when",
    [datetime(2026, 10, 21, 13, 5, 7), datetime(2026, 10, 21, 9, 5, 7),
     datetime(2026, 10, 21, 0, 15, 0), datetime(2026, 10, 21, 12, 0, 59)],
)
def test_round_trip(tmp_path, when):
    run = make_run(when=when)
    back = DADRun.from_chromeleon(run.to_chromeleon(tmp_path / "run.txt"))

    assert np.abs(back.intensity - run.intensity).max() <= 5e-7 + 1e-9   # 6-decimal rounding
    assert np.abs(back.time - run.time).max() <= 5e-7 + 1e-9
    assert np.array_equal(back.wl, run.wl)
    for field in EXPORTED_FIELDS:
        assert getattr(back.meta, field) == getattr(run.meta, field), field


def test_layout_matches_chromeleon(tmp_path):
    path = make_run().to_chromeleon(tmp_path / "run.txt")
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbfURL\t")        # UTF-8 byte order mark
    text = raw.decode("utf-8-sig")
    assert "\r\n" in text and "\n" not in text.replace("\r\n", "")
    assert "Inject Time\t10/21/2026 1:05:07 PM +02:00\r\n" in text   # summer time
    assert "\t-1,234.567890\t" in text
    assert "\r\nRaw Data:\r\nTime (min)\tIntegr.Time (s)\t190.0\t192.0\t" in text


def test_missing_metadata_is_left_empty(tmp_path):
    run = make_run()
    bare = DADRun(run.time, run.wl, run.intensity, DADMeta(
        sample_name="X", sequence="S", injection_datetime=datetime(2026, 1, 2, 3, 4, 5)))
    back = DADRun.from_chromeleon(bare.to_chromeleon(tmp_path / "bare.txt"))
    assert back.meta.injection_number is None
    assert back.meta.injection_volume_ul is None
    assert back.meta.vial_position is None
    assert back.meta.injection_id is None


def test_refuses_to_overwrite(tmp_path):
    run = make_run()
    path = run.to_chromeleon(tmp_path / "run.txt")
    with pytest.raises(FileExistsError):
        run.to_chromeleon(path)
    run.to_chromeleon(path, overwrite=True)


def test_sequence_names_repeated_samples(tmp_path):
    seq = DADSequence([make_run("A", "1.smp", 1), make_run("A", "2.smp", 2), make_run("B", "3.smp", 3)])
    paths = seq.to_chromeleon(tmp_path)
    assert sorted(p.name for p in paths) == ["A (1.smp).txt", "A (2.smp).txt", "B.txt"]
    assert len(DADSequence.from_chromeleon(tmp_path)) == 3


@pytest.mark.parametrize(
    "when, timezone, expected",
    [
        (datetime(2026, 11, 2, 21, 54, 20), "Europe/Copenhagen", "11/2/2026 9:54:20 PM +01:00"),
        (datetime(2026, 7, 1, 9, 0, 0), "Europe/Copenhagen", "7/1/2026 9:00:00 AM +02:00"),
        (datetime(2026, 7, 1, 9, 0, 0), "UTC", "7/1/2026 9:00:00 AM +00:00"),
        (datetime(2026, 7, 1, 9, 0, 0), "America/New_York", "7/1/2026 9:00:00 AM -04:00"),
    ],
)
def test_inject_time_has_utc_offset(tmp_path, when, timezone, expected):
    path = make_run(when=when).to_chromeleon(tmp_path / "run.txt", timezone=timezone)
    # read_bytes, not read_text: read_text would turn the file's \r\n into \n
    assert f"Inject Time\t{expected}\r\n" in path.read_bytes().decode("utf-8-sig")


def test_unknown_timezone(tmp_path):
    with pytest.raises(ValueError, match="time zone"):
        make_run().to_chromeleon(tmp_path / "run.txt", timezone="Mars/Olympus")
