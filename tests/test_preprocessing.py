from datetime import datetime

import numpy as np
import pytest

from hplcdad import Chromatogram, ChromatogramSet, DADMeta, Spectrum
from hplcdad.preprocessing import asls_baseline, savgol_smooth

META = DADMeta(sample_name="synthetic", sequence="tests", injection_datetime=datetime(2026, 1, 1))


def synthetic_chromatogram(seed=0):
    """Three Gaussian peaks on a curved baseline, with noise, at 2.5 Hz over 36 min."""
    time = np.linspace(0, 36, 5401)
    baseline = 5 + 0.3 * time + 2 * np.sin(time / 12)
    peaks = sum(
        height * np.exp(-0.5 * ((time - centre) / width) ** 2)
        for height, centre, width in [(100, 8, 0.08), (60, 15, 0.10), (150, 22, 0.12)]
    )
    noise = np.random.default_rng(seed).normal(0, 0.05, time.size)
    return time, baseline, baseline + peaks + noise


def test_asls_recovers_known_baseline():
    time, baseline, y = synthetic_chromatogram()
    estimate = asls_baseline(y, lam=1e7, p=0.01)
    assert np.abs(estimate - baseline).max() < 0.01 * 150   # within 1% of the tallest peak


def test_savgol_keeps_a_quadratic_unchanged():
    x = np.linspace(0, 10, 501)
    y = 1 + 2 * x + 0.5 * x**2
    smooth = savgol_smooth(y, x, 25, polyorder=2)
    assert np.allclose(smooth, y, atol=1e-9)


def test_savgol_first_derivative():
    x = np.linspace(0, 10, 501)
    y = 2 * x + 0.5 * x**2
    derivative = savgol_smooth(y, x, 25, polyorder=2, deriv=1)
    assert np.allclose(derivative, 2 + x, atol=1e-8)


def test_savgol_rejects_uneven_axis():
    x = np.array([0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0])
    with pytest.raises(ValueError, match="evenly spaced"):
        savgol_smooth(np.zeros_like(x), x, 3)


def test_savgol_rejects_too_short_window():
    x = np.linspace(0, 10, 101)
    with pytest.raises(ValueError, match="polyorder"):
        savgol_smooth(np.zeros_like(x), x, 3, polyorder=3)


def test_savgol_rejects_even_window():
    x = np.linspace(0, 10, 101)
    with pytest.raises(ValueError, match="odd"):
        savgol_smooth(np.zeros_like(x), x, 10)


def test_chromatogram_methods_return_new_objects_with_history():
    time, _, y = synthetic_chromatogram()
    original = Chromatogram(wavelength=270.0, time=time, intensity=y, meta=META)
    processed = original.correct_baseline(lam=1e7).savgol(11)

    assert processed is not original
    assert np.array_equal(original.intensity, y)
    assert original.processing == ()
    assert len(processed.processing) == 2
    assert processed.processing[0].startswith("asls baseline correction")


def test_sets_apply_to_every_trace():
    time, _, y = synthetic_chromatogram()
    chromatograms = ChromatogramSet(
        [Chromatogram(wavelength=270.0, time=time, intensity=y, meta=META)] * 3, run_index=[4, 5, 6]
    )
    corrected = chromatograms.correct_baseline(lam=1e7)
    assert len(corrected) == 3
    assert corrected.run_index == (4, 5, 6)


def test_spectrum_savgol():
    wl = np.arange(190.0, 642.0, 2.0)
    spectrum = Spectrum(time=10.0, wl=wl, intensity=np.exp(-((wl - 270) / 20) ** 2), meta=META)
    smooth = spectrum.savgol(5)
    assert smooth.intensity.shape == wl.shape
    assert "5 points" in smooth.processing[0]
