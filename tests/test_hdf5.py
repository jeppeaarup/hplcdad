from datetime import datetime

import numpy as np
import pytest

from hplcdad import DADMeta, DADRun

h5py = pytest.importorskip("h5py")

META = DADMeta(sample_name="synthetic", sequence="tests", injection_datetime=datetime(2026, 1, 1))


def synthetic_run():
    time = np.linspace(0, 5, 11)
    wavelength = np.arange(200.0, 220.0, 2.0)
    return DADRun(time, wavelength, np.random.default_rng(0).random((len(time), len(wavelength))), META)


def test_hdf5_round_trip_uses_new_names(tmp_path):
    run = synthetic_run()
    path = run.to_hdf5(tmp_path / "run.h5")
    with h5py.File(path, "r") as file:
        assert {"time", "wavelength", "intensity"} <= set(file)
    back = DADRun.from_hdf5(path)
    assert np.array_equal(back.time, run.time)
    assert np.array_equal(back.intensity, run.intensity)


@pytest.mark.parametrize(
    "version, names",
    [(1, ("rt", "wl", "signal")), (2, ("time", "wl", "intensity"))],
)
def test_older_hdf5_schemas_still_load(tmp_path, version, names):
    run = synthetic_run()
    path = tmp_path / "old.h5"
    with h5py.File(path, "w") as file:
        file.attrs["hplcdad_schema_version"] = version
        file.attrs["sample_name"] = META.sample_name
        file.attrs["sequence"] = META.sequence
        file.attrs["injection_datetime"] = META.injection_datetime.isoformat()
        file.create_dataset(names[0], data=run.time).attrs["units"] = "min"
        file.create_dataset(names[1], data=run.wavelength)
        file.create_dataset(names[2], data=run.intensity).attrs["units"] = "mAU"
    back = DADRun.from_hdf5(path)
    assert np.array_equal(back.time, run.time)
    assert np.array_equal(back.wavelength, run.wavelength)
    assert np.array_equal(back.intensity, run.intensity)


def test_short_getter_names_are_aliases():
    run = synthetic_run()
    assert np.array_equal(run.chromatogram(204).intensity, run.get_chromatogram(204).intensity)
    assert np.array_equal(run.spectrum(2.0).intensity, run.get_spectrum(2.0).intensity)
