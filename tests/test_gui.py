import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from hplcdad import DADMeta, DADRun, DADSequence
from hplcdad.gui import Cancelled, default_output_folder, detect_format, load, save
from hplcdad.readers.chromeleon_cmbx import count_cmbx_fields


def make_run(number):
    time = np.arange(50) * 0.01
    wl = np.arange(200.0, 220.0, 2.0)
    intensity = np.random.default_rng(number).normal(0, 10, (len(time), len(wl)))
    meta = DADMeta(
        sample_name=f"Brew {number}",
        sequence="Test sequence",
        injection_datetime=datetime(2026, 10, 21, 13, number, 0),
        injection_number=number,
    )
    return DADRun(time, wl, intensity, meta)


@pytest.fixture
def export_folder(tmp_path):
    folder = tmp_path / "exports"
    DADSequence([make_run(i) for i in (1, 2, 3)]).to_chromeleon(folder)
    return folder


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, done, total):
        self.calls.append((done, total))


def test_detect_format(tmp_path, export_folder):
    assert detect_format(export_folder) == "chromeleon"
    assert detect_format(next(export_folder.glob("*.txt"))) == "chromeleon"
    (tmp_path / "a.cmbx").write_bytes(b"")
    assert detect_format(tmp_path / "a.cmbx") == "cmbx"
    (tmp_path / "notes.docx").write_bytes(b"")
    assert detect_format(tmp_path / "notes.docx") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert detect_format(empty) is None


def test_default_output_folder(tmp_path, export_folder):
    assert default_output_folder(export_folder, "hdf5") == tmp_path / "exports (h5)"
    archive = tmp_path / "Spike.cmbx"
    archive.write_bytes(b"")
    assert default_output_folder(archive, "chromeleon") == tmp_path / "Spike (txt)"
    one_file = next(export_folder.glob("*.txt"))
    assert default_output_folder(one_file, "hdf5") == export_folder


def test_text_to_hdf5_and_back_with_progress(tmp_path, export_folder):
    pytest.importorskip("h5py")
    reading = Recorder()
    seq = load(export_folder, "chromeleon", progress=reading)
    assert reading.calls == [(0, 3), (1, 3), (2, 3), (3, 3)]

    writing = Recorder()
    paths = save(seq, tmp_path / "h5", "hdf5", progress=writing)
    assert len(paths) == 3
    assert writing.calls == [(0, 3), (1, 3), (2, 3), (3, 3)]

    back = load(tmp_path / "h5", "hdf5")
    paths = save(back, tmp_path / "txt", "chromeleon", timezone="UTC")
    assert sorted(p.name for p in paths) == ["Brew 1.txt", "Brew 2.txt", "Brew 3.txt"]
    again = load(tmp_path / "txt", "chromeleon")
    np.testing.assert_allclose(again[0].intensity, seq[0].intensity, atol=1e-6)


def test_save_refuses_to_overwrite_unless_asked(tmp_path, export_folder):
    seq = load(export_folder, "chromeleon")
    save(seq, tmp_path / "out", "chromeleon")
    with pytest.raises(FileExistsError):
        save(seq, tmp_path / "out", "chromeleon")
    save(seq, tmp_path / "out", "chromeleon", overwrite=True)


def test_cancel_from_progress_stops_writing(tmp_path, export_folder):
    seq = load(export_folder, "chromeleon")

    def cancel_after_first(done, total):
        if done == 1:
            raise Cancelled

    with pytest.raises(Cancelled):
        save(seq, tmp_path / "out", "chromeleon", progress=cancel_after_first)
    assert len(list((tmp_path / "out").glob("*.txt"))) == 1


def test_count_cmbx_fields_reads_only_the_header(tmp_path):
    header = """<root>
      <ChromeleonElement ItemType="Dionex.Chromeleon.Data.Sequence" Name="Seq"/>
      <ChromeleonElement ItemType="Dionex.Chromeleon.Data.Injection" Name="A">
        <Intensity RawDataFilename="a.sfd"/>
      </ChromeleonElement>
      <ChromeleonElement ItemType="Dionex.Chromeleon.Data.Injection" Name="Wash"/>
      <ChromeleonElement ItemType="Dionex.Chromeleon.Data.Injection" Name="B">
        <Intensity RawDataFilename="b.sfd"/>
      </ChromeleonElement>
    </root>"""
    path = tmp_path / "test.cmbx"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("header.xml", header)
    assert count_cmbx_fields(path) == 2

    not_zip = tmp_path / "bad.cmbx"
    not_zip.write_text("hello")
    with pytest.raises(ValueError, match="not a Chromeleon .cmbx archive"):
        count_cmbx_fields(not_zip)


def test_window_opens_and_suggests_output(export_folder):
    tk = pytest.importorskip("tkinter")
    from hplcdad.gui import ConverterApp

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    try:
        app = ConverterApp(root)
        app.source_var.set(str(export_folder))
        assert app.input_format() == "chromeleon"
        assert Path(app.output_var.get()) == default_output_folder(export_folder, "hdf5")
        assert str(app.timezone_box.cget("state")) == "disabled"

        app.output_format_var.set("chromeleon")
        app.on_output_format_changed()
        assert Path(app.output_var.get()) == default_output_folder(export_folder, "chromeleon")
        assert str(app.timezone_box.cget("state")) == "normal"
    finally:
        root.destroy()
