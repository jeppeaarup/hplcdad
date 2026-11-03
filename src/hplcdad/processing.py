import numpy as np
from .raw import RawRun, TIME_UNITS_TO_MIN, SIGNAL_UNITS_TO_MAU

def orient_matrix(rt, wl, signal, *, rt_as_rows=None):
    """Return signal as time x wavelength, transposing if needed.

    rt_as_rows: True if time runs along the rows, False if wavelength does,
    None if unknown. Only needed to resolve square matrices.
    """
    rt = np.asarray(rt)
    wl = np.asarray(wl)
    signal = np.asarray(signal)

    if signal.ndim != 2:
        raise ValueError(
            f"Signal must be 2-dimensional, got {signal.ndim} dimension(s) "
            f"with shape {signal.shape}."
        )

    n_time = len(rt)
    n_wl = len(wl)
    shape = signal.shape

    matches_as_is = shape == (n_time, n_wl)
    matches_transposed = shape == (n_wl, n_time)

    if not matches_as_is and not matches_transposed:
        raise ValueError(
            f"Signal shape {shape} fits neither (time, wavelength) = "
            f"({n_time}, {n_wl}) nor (wavelength, time) = ({n_wl}, {n_time})."
        )

    if matches_as_is and matches_transposed:
        # Square matrix: the shape cannot tell the orientation, so the hint must.
        if rt_as_rows is None:
            raise ValueError(
                f"Signal is square {shape} (same number of time points and "
                f"wavelengths), so its orientation is ambiguous and no "
                f"rt_as_rows hint was given."
            )
        return signal if rt_as_rows else signal.T

    if matches_as_is:
        if rt_as_rows is False:
            raise ValueError(
                f"Signal shape {shape} says time runs along the rows, but "
                f"rt_as_rows=False says wavelength does. Check the reader."
            )
        return signal

    if rt_as_rows is True:
        raise ValueError(
            f"Signal shape {shape} says wavelength runs along the rows, but "
            f"rt_as_rows=True says time does. Check the reader."
        )
    return signal.T


def convert_units(rt, signal, time_unit, signal_unit):
    """Return time in minutes and signal in mAU, as new arrays."""
    if time_unit not in TIME_UNITS_TO_MIN:
        raise ValueError(
            f"Unknown time unit {time_unit!r}. Known units: {list(TIME_UNITS_TO_MIN)}."
        )
    if signal_unit not in SIGNAL_UNITS_TO_MAU:
        raise ValueError(
            f"Unknown signal unit {signal_unit!r}. Known units: {list(SIGNAL_UNITS_TO_MAU)}."
        )
    rt_min = np.asarray(rt) * TIME_UNITS_TO_MIN[time_unit]
    signal_mau = np.asarray(signal) * SIGNAL_UNITS_TO_MAU[signal_unit]
    return rt_min, signal_mau


def validate(rt, wl, signal):
    """Raise ValueError unless the axes and signal form a consistent run."""
    rt = np.asarray(rt)
    wl = np.asarray(wl)
    signal = np.asarray(signal)

    if rt.ndim != 1 or wl.ndim != 1:
        raise ValueError(
            f"Axes must be 1-dimensional, got rt with shape {rt.shape} "
            f"and wl with shape {wl.shape}."
        )
    if len(rt) == 0 or len(wl) == 0:
        raise ValueError("Time and wavelength axes must not be empty.")
    if signal.shape != (len(rt), len(wl)):
        raise ValueError(
            f"Signal shape {signal.shape} does not match the axes: "
            f"expected ({len(rt)}, {len(wl)}) = (time, wavelength)."
        )
    if not (np.isfinite(rt).all() and np.isfinite(wl).all()):
        raise ValueError("Axes contain NaN or infinite values.")
    if np.any(np.diff(rt) <= 0):
        raise ValueError("Time axis must be strictly increasing.")
    if np.any(np.diff(wl) <= 0):
        raise ValueError("Wavelength axis must be strictly increasing.")
    n_bad = signal.size - np.count_nonzero(np.isfinite(signal))
    if n_bad:
        raise ValueError(f"Signal contains {n_bad} NaN or infinite value(s).")


def normalise(raw: RawRun):
    """Clean a RawRun; return (rt, wl, signal, meta) ready for DADRun."""
    rt, signal = convert_units(raw.rt, raw.signal, raw.time_unit, raw.signal_unit)
    wl = np.asarray(raw.wl)
    signal = orient_matrix(rt, wl, signal, rt_as_rows=raw.rt_as_rows)
    validate(rt, wl, signal)
    return rt, wl, signal, raw.meta