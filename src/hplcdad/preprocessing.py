"""Signal processing for 1-D traces: AsLS baseline and Savitzky-Golay filter.

Plain array functions; Chromatogram, Spectrum and the trace sets call them.
"""
import numpy as np
from pybaselines import Baseline
from scipy.signal import savgol_filter


def asls_baseline(y, *, lam=1e6, p=1e-2, n_iter=50):
    """Estimate a baseline with asymmetric least squares (Eilers & Boelens, 2005).

    Uses the AsLS implementation in pybaselines. It fits a smooth curve z
    to y by minimising

        sum(w * (y - z)**2) + lam * sum(diff(z, 2)**2)

    where points above the curve get weight p and points below it 1 - p.
    With a small p the curve follows the lower edge of the intensity, so peaks
    are left out of the baseline.

    Parameters
    ----------
    y : 1-D array
        The intensity.
    lam : float, default 1e6
        Smoothness; larger gives a stiffer baseline. Typical values are
        1e4 to 1e9. The right value depends on the number of points, so
        check the result after changing the data rate or cropping.
    p : float, default 0.01
        Asymmetry, between 0 and 1. Typical values are 0.001 to 0.05.
    n_iter : int, default 10
        Largest number of reweighting steps (pybaselines' ``max_iter``).
        Stops early when the baseline changes by less than pybaselines'
        default tolerance between steps.

    Returns
    -------
    1-D array
        The baseline, same length as y.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError(f"Intensity must be 1-dimensional, got shape {y.shape}.")
    if len(y) < 3:
        raise ValueError(f"Need at least 3 points for a baseline, got {len(y)}.")
    if not 0 < p < 1:
        raise ValueError(f"p must be between 0 and 1, got {p}.")
    if lam <= 0:
        raise ValueError(f"lam must be positive, got {lam}.")
    if n_iter < 1:
        raise ValueError(f"n_iter must be at least 1, got {n_iter}.")

    baseline, _ = Baseline().asls(y, lam=lam, p=p, max_iter=n_iter)
    return baseline


def savgol_smooth(y, x, window_length, *, polyorder=2, deriv=0, mode="interp", unit=""):
    """Savitzky-Golay filter.

    Parameters
    ----------
    y : 1-D array
        The intensity.
    x : 1-D array
        Its axis (time in min or wavelength in nm). Must be evenly spaced;
        the spacing is used to scale derivatives.
    window_length : int
        Window length in points; must be odd and larger than polyorder.
    polyorder : int, default 2
        Order of the fitted polynomial.
    deriv : int, default 0
        Derivative order. Derivatives are per unit of x (e.g. mAU/min).
    mode : str, default "interp"
        Edge handling, passed on to scipy.signal.savgol_filter.
    unit : str
        Unit of x, used in error messages.

    Returns
    -------
    1-D array
        The filtered intensity.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or y.shape != x.shape:
        raise ValueError(f"Intensity {y.shape} and axis {x.shape} must be 1-D and the same length.")
    if len(x) < 2:
        raise ValueError("Need at least 2 points.")

    steps = np.diff(x)
    step = float(np.median(steps))
    if not np.allclose(steps, step, rtol=1e-3, atol=0):
        raise ValueError(
            f"The axis is not evenly spaced (steps from {steps.min():g} to "
            f"{steps.max():g} {unit}); the Savitzky-Golay filter needs equal spacing."
        )
    if int(window_length) != window_length or window_length < 1:
        raise ValueError(f"window_length must be a positive whole number of points, got {window_length}.")
    window_length = int(window_length)
    if window_length % 2 == 0:
        raise ValueError(f"window_length must be odd, got {window_length}.")
    if window_length <= polyorder:
        raise ValueError(
            f"window_length ({window_length} points) must be larger than polyorder ({polyorder})."
        )
    if window_length > len(y):
        raise ValueError(
            f"window_length ({window_length} points) is longer than the trace ({len(y)} points)."
        )
    if deriv < 0 or deriv > polyorder:
        raise ValueError(f"deriv must be between 0 and polyorder ({polyorder}), got {deriv}.")

    return savgol_filter(y, window_length, polyorder, deriv=deriv, delta=step, mode=mode)
