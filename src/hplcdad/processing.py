import numpy as np
from .raw import RawRun, TIME_UNITS_TO_MIN, INTENSITY_UNITS_TO_MAU

def orient_matrix(time, wavelength, intensity, *, time_as_rows=None):
    """Return intensity as time x wavelength, transposing if needed.

    time_as_rows: True if time runs along the rows, False if wavelength does,
    None if unknown. Only needed to resolve square matrices.
    """
    time = np.asarray(time)
    wavelength = np.asarray(wavelength)
    intensity = np.asarray(intensity)

    if intensity.ndim != 2:
        raise ValueError(
            f"Intensity must be 2-dimensional, got {intensity.ndim} dimension(s) "
            f"with shape {intensity.shape}."
        )

    n_time = len(time)
    n_wavelength = len(wavelength)
    shape = intensity.shape

    matches_as_is = shape == (n_time, n_wavelength)
    matches_transposed = shape == (n_wavelength, n_time)

    if not matches_as_is and not matches_transposed:
        raise ValueError(
            f"Intensity shape {shape} fits neither (time, wavelength) = "
            f"({n_time}, {n_wavelength}) nor (wavelength, time) = ({n_wavelength}, {n_time})."
        )

    if matches_as_is and matches_transposed:
        # Square matrix: the shape cannot tell the orientation, so the hint must.
        if time_as_rows is None:
            raise ValueError(
                f"Intensity is square {shape} (same number of time points and "
                f"wavelengths), so its orientation is ambiguous and no "
                f"time_as_rows hint was given."
            )
        return intensity if time_as_rows else intensity.T

    if matches_as_is:
        if time_as_rows is False:
            raise ValueError(
                f"Intensity shape {shape} says time runs along the rows, but "
                f"time_as_rows=False says wavelength does. Check the reader."
            )
        return intensity

    if time_as_rows is True:
        raise ValueError(
            f"Intensity shape {shape} says wavelength runs along the rows, but "
            f"time_as_rows=True says time does. Check the reader."
        )
    return intensity.T


def convert_units(time, intensity, time_unit, intensity_unit):
    """Return time in minutes and intensity in mAU, as new arrays."""
    if time_unit not in TIME_UNITS_TO_MIN:
        raise ValueError(
            f"Unknown time unit {time_unit!r}. Known units: {list(TIME_UNITS_TO_MIN)}."
        )
    if intensity_unit not in INTENSITY_UNITS_TO_MAU:
        raise ValueError(
            f"Unknown intensity unit {intensity_unit!r}. Known units: {list(INTENSITY_UNITS_TO_MAU)}."
        )
    time_min = np.asarray(time) * TIME_UNITS_TO_MIN[time_unit]
    intensity_mau = np.asarray(intensity) * INTENSITY_UNITS_TO_MAU[intensity_unit]
    return time_min, intensity_mau


def validate(time, wavelength, intensity):
    """Raise ValueError unless the axes and intensity form a consistent run."""
    time = np.asarray(time)
    wavelength = np.asarray(wavelength)
    intensity = np.asarray(intensity)

    if time.ndim != 1 or wavelength.ndim != 1:
        raise ValueError(
            f"Axes must be 1-dimensional, got time with shape {time.shape} "
            f"and wavelength with shape {wavelength.shape}."
        )
    if len(time) == 0 or len(wavelength) == 0:
        raise ValueError("Time and wavelength axes must not be empty.")
    if intensity.shape != (len(time), len(wavelength)):
        raise ValueError(
            f"Intensity shape {intensity.shape} does not match the axes: "
            f"expected ({len(time)}, {len(wavelength)}) = (time, wavelength)."
        )
    if not (np.isfinite(time).all() and np.isfinite(wavelength).all()):
        raise ValueError("Axes contain NaN or infinite values.")
    if np.any(np.diff(time) <= 0):
        raise ValueError("Time axis must be strictly increasing.")
    if np.any(np.diff(wavelength) <= 0):
        raise ValueError("Wavelength axis must be strictly increasing.")
    n_bad = intensity.size - np.count_nonzero(np.isfinite(intensity))
    if n_bad:
        raise ValueError(f"Intensity contains {n_bad} NaN or infinite value(s).")


def normalise(raw: RawRun):
    """Clean a RawRun; return (time, wavelength, intensity, meta) ready for DADRun."""
    time, intensity = convert_units(raw.time, raw.intensity, raw.time_unit, raw.intensity_unit)
    wavelength = np.asarray(raw.wavelength)
    intensity = orient_matrix(time, wavelength, intensity, time_as_rows=raw.time_as_rows)
    validate(time, wavelength, intensity)
    return time, wavelength, intensity, raw.meta