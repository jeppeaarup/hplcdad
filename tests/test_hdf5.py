from datetime import datetime

import numpy as np
import pytest

from hplcdad import DADMeta, DADRun

h5py = pytest.importorskip("h5py")

META = DADMeta(sample_name="synthetic", sequence="tests", injection_datetime=datetime(2026, 1, 1))


def synthetic_run():
    time = np.linspace(0, 5, 11)
    wl = np.arange(200.0, 220.0, 2.0)
    return DADRun(time, wl, np.random.default_rng(0).random((len(time), len(wl))), META)


def test_hdf5_round_trip_uses_new_names(tmp_path):
    run = synthetic_run()
    path = run.to_hdf5(tmp_path / "run.h5")
    with h5py.File(path, "r") as file:
        assert {"time", "wl", "intensity"} <= set(file)
    back = DADRun.from_hdf5(path)
    assert np.array_equal(back.time, run.time)
    assert np.array_equal(back.intensity, run.intensity)


def test_hdf5_schema_1_files_with_rt_and_signal_still_load(tmp_path):
    run = synthetic_run()
    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as file:
        file.attrs["hplcdad_schema_version"] = 1
        file.attrs["sample_name"] = META.sample_name
        file.attrs["sequence"] = META.sequence
        file.attrs["injection_datetime"] = META.injection_datetime.isoformat()
        file.create_dataset("rt", data=run.time).attrs["units"] = "min"
        file.create_dataset("wl", data=run.wl)
        file.create_dataset("signal", data=run.intensity).attrs["units"] = "mAU"
    back = DADRun.from_hdf5(path)
    assert np.array_equal(back.time, run.time)
    assert np.array_equal(back.intensity, run.intensity)
