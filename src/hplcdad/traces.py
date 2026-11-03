from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .meta import DADMeta
from .preprocessing import asls_baseline, savgol_smooth


@dataclass(frozen=True)
class Spectrum:
    """A spectrum extracted from a DAD run: signal vs wavelength at one time."""

    time: float          # time point used, in min
    wl: np.ndarray       # wavelength axis, in nm
    signal: np.ndarray   # mAU
    meta: DADMeta
    requested_time: float | None = None   # time asked for, in min
    processing: tuple[str, ...] = ()      # steps applied, in order

    def __post_init__(self):
        if np.shape(self.signal) != np.shape(self.wl):
            raise ValueError(
                f"Signal shape {np.shape(self.signal)} does not match "
                f"wavelength axis shape {np.shape(self.wl)}."
            )

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (wl, signal) as copies."""
        return self.wl.copy(), self.signal.copy()

    def to_pandas(self) -> pd.Series:
        """Return the signal as a Series indexed by wavelength (nm)."""
        return pd.Series(
            self.signal.copy(),
            index=pd.Index(self.wl.copy(), name="wl_nm"),
            name=self.meta.sample_name,
        )

    def savgol(self, window: float, *, polyorder: int = 2, deriv: int = 0) -> "Spectrum":
        """Return a new Spectrum smoothed with a Savitzky-Golay filter along wavelength.

        Parameters
        ----------
        window : float
            Window width in nm, rounded to the nearest odd number of points.
        polyorder : int, default 2
            Order of the fitted polynomial.
        deriv : int, default 0
            Derivative order; 1 or 2 give spectral derivatives (mAU/nm, mAU/nm²).

        Examples
        --------
        >>> smooth = spectrum.savgol(10)
        >>> first_derivative = spectrum.savgol(10, deriv=1)
        """
        signal, n_points = savgol_smooth(
            self.signal, self.wl, window, polyorder=polyorder, deriv=deriv, unit="nm"
        )
        step = (
            f"savgol (window={window:g} nm = {n_points} points, "
            f"polyorder={polyorder}, deriv={deriv})"
        )
        return replace(self, signal=signal, processing=self.processing + (step,))


@dataclass(frozen=True)
class Chromatogram:
    """A chromatogram extracted from a DAD run: signal vs time at one wavelength."""

    wavelength: float    # wavelength used, in nm
    rt: np.ndarray       # time axis, in min
    signal: np.ndarray   # mAU
    meta: DADMeta
    requested_wavelength: float | None = None   # wavelength asked for, in nm
    processing: tuple[str, ...] = ()            # steps applied, in order

    def __post_init__(self):
        if np.shape(self.signal) != np.shape(self.rt):
            raise ValueError(
                f"Signal shape {np.shape(self.signal)} does not match "
                f"time axis shape {np.shape(self.rt)}."
            )

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (rt, signal) as copies."""
        return self.rt.copy(), self.signal.copy()

    def to_pandas(self) -> pd.Series:
        """Return the signal as a Series indexed by time (min)."""
        return pd.Series(
            self.signal.copy(),
            index=pd.Index(self.rt.copy(), name="rt_min"),
            name=self.meta.sample_name,
        )

    def estimate_baseline(
        self, *, lam: float = 1e6, p: float = 0.01, n_iter: int = 10
    ) -> "Chromatogram":
        """Return the baseline as a new Chromatogram, estimated with asymmetric least squares.

        Useful for plotting the baseline over the data to check the settings.
        See ``correct_baseline`` for the parameters.

        Examples
        --------
        >>> baseline = chromatogram.estimate_baseline(lam=1e7)
        """
        baseline = asls_baseline(self.signal, lam=lam, p=p, n_iter=n_iter)
        step = f"asls baseline estimate (lam={lam:g}, p={p:g})"
        return replace(self, signal=baseline, processing=self.processing + (step,))

    def correct_baseline(
        self, *, lam: float = 1e6, p: float = 0.01, n_iter: int = 10
    ) -> "Chromatogram":
        """Return a new Chromatogram with the baseline removed (asymmetric least squares).

        Fits a smooth curve along the lower edge of the signal, leaving out
        the peaks, and subtracts it (Eilers & Boelens, 2005).

        Parameters
        ----------
        lam : float, default 1e6
            Smoothness; larger gives a stiffer baseline. Typical values are
            1e4 to 1e9; the right value depends on the number of points.
        p : float, default 0.01
            Asymmetry, between 0 and 1; smaller keeps the baseline further
            below the peaks. Typical values are 0.001 to 0.05.
        n_iter : int, default 10
            Largest number of reweighting steps.

        Examples
        --------
        >>> corrected = chromatogram.correct_baseline(lam=1e7, p=0.01)
        """
        baseline = asls_baseline(self.signal, lam=lam, p=p, n_iter=n_iter)
        step = f"asls baseline correction (lam={lam:g}, p={p:g})"
        return replace(self, signal=self.signal - baseline, processing=self.processing + (step,))

    def savgol(self, window: float, *, polyorder: int = 2, deriv: int = 0) -> "Chromatogram":
        """Return a new Chromatogram smoothed with a Savitzky-Golay filter along time.

        Parameters
        ----------
        window : float
            Window width in min, rounded to the nearest odd number of points.
            Keep it well below the peak width, or peaks get flattened.
        polyorder : int, default 2
            Order of the fitted polynomial.
        deriv : int, default 0
            Derivative order (mAU/min for 1).

        Examples
        --------
        >>> smooth = chromatogram.savgol(0.05)
        """
        signal, n_points = savgol_smooth(
            self.signal, self.rt, window, polyorder=polyorder, deriv=deriv, unit="min"
        )
        step = (
            f"savgol (window={window:g} min = {n_points} points, "
            f"polyorder={polyorder}, deriv={deriv})"
        )
        return replace(self, signal=signal, processing=self.processing + (step,))


class _TraceSet:
    """Shared behaviour of ChromatogramSet and SpectrumSet.

    Holds one trace per run, together with each run's position in the
    DADSequence it came from, so labels survive slicing.
    """

    _axis = ""        # "rt" or "wl": attribute name of the x axis
    _axis_label = ""  # column index name in to_pandas()
    _unit = ""

    def __init__(self, traces: Iterable, run_index: Iterable[int] | None = None):
        traces = tuple(traces)
        run_index = (
            tuple(range(len(traces)))
            if run_index is None
            else tuple(int(i) for i in run_index)
        )
        if len(run_index) != len(traces):
            raise ValueError(
                f"Got {len(traces)} traces but {len(run_index)} run numbers."
            )
        self._traces = traces
        self._run_index = run_index

    def __len__(self) -> int:
        return len(self._traces)

    def __iter__(self) -> Iterator:
        return iter(self._traces)

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_)):
            return self._traces[key]
        if isinstance(key, slice):
            return type(self)(self._traces[key], self._run_index[key])
        raise TypeError(f"Index a {type(self).__name__} with an int or a slice.")

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self)} traces)"

    @property
    def run_index(self) -> tuple[int, ...]:
        """Position of each trace's run in the DADSequence it came from."""
        return self._run_index

    @property
    def names(self) -> list[str]:
        """Sample names, in order."""
        return [trace.meta.sample_name for trace in self._traces]

    def _apply(self, method: str, *args, **kwargs):
        """Call a method on every trace and return a new set with the results."""
        return type(self)(
            [getattr(trace, method)(*args, **kwargs) for trace in self._traces],
            self._run_index,
        )

    def _index(self) -> pd.MultiIndex:
        return pd.MultiIndex.from_arrays(
            [list(self._run_index), self.names], names=["run", "sample_name"]
        )

    def _to_numpy(self, grid, atol: float) -> tuple[np.ndarray, np.ndarray]:
        labels = [f"Run {i} ({name!r})" for i, name in zip(self._run_index, self.names)]
        return align_traces(
            [getattr(t, self._axis) for t in self._traces],
            [t.signal for t in self._traces],
            grid,
            atol,
            name=self._axis,
            unit=self._unit,
            labels=labels,
        )

    def _to_pandas(self, grid, atol: float) -> pd.DataFrame:
        axis, X = self._to_numpy(grid, atol)
        return pd.DataFrame(
            X, index=self._index(), columns=pd.Index(axis, name=self._axis_label)
        )


class ChromatogramSet(_TraceSet):
    """Chromatograms at one wavelength from several runs.

    Behaves like a read-only list of Chromatogram objects, and converts to a
    runs x time matrix with ``to_numpy`` or ``to_pandas``. Rows are labelled
    by (run, sample_name).
    """

    _axis = "rt"
    _axis_label = "rt_min"
    _unit = "min"

    def _check_wavelength(self):
        used = sorted({t.wavelength for t in self._traces})
        if len(used) > 1:
            raise ValueError(
                f"The chromatograms were taken at different wavelengths {used} nm, "
                f"because the runs have different wavelength axes."
            )

    def estimate_baseline(self, *, lam: float = 1e6, p: float = 0.01, n_iter: int = 10) -> "ChromatogramSet":
        """Return the baseline of every chromatogram; see ``Chromatogram.correct_baseline``."""
        return self._apply("estimate_baseline", lam=lam, p=p, n_iter=n_iter)

    def correct_baseline(self, *, lam: float = 1e6, p: float = 0.01, n_iter: int = 10) -> "ChromatogramSet":
        """Remove the baseline from every chromatogram; see ``Chromatogram.correct_baseline``.

        Examples
        --------
        >>> corrected = seq.get_chromatogram(270).correct_baseline(lam=1e7)
        """
        return self._apply("correct_baseline", lam=lam, p=p, n_iter=n_iter)

    def savgol(self, window: float, *, polyorder: int = 2, deriv: int = 0) -> "ChromatogramSet":
        """Smooth every chromatogram; see ``Chromatogram.savgol`` (window in min)."""
        return self._apply("savgol", window, polyorder=polyorder, deriv=deriv)

    def to_numpy(self, *, rt_grid=None, rt_atol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        """Return (rt, X) with X of shape (n_runs, n_time), in mAU.

        Parameters
        ----------
        rt_grid : None, "overlap", or array, default None
            None requires every chromatogram to have the same time axis.
            "overlap" interpolates onto the first run's time points inside the
            range covered by every run. An array interpolates onto those
            times (min), which must lie inside every run.
        rt_atol : float, default 1e-6
            Tolerance in minutes when checking that the time axes match.
        """
        self._check_wavelength()
        return self._to_numpy(rt_grid, rt_atol)

    def to_pandas(self, *, rt_grid=None, rt_atol: float = 1e-6) -> pd.DataFrame:
        """Return a DataFrame with one row per run and one column per time (min).

        Rows are indexed by (run, sample_name); see ``to_numpy`` for options.

        Examples
        --------
        >>> df = seq.get_chromatogram(314, rt_min=5, rt_max=40).to_pandas()
        >>> df.xs("QC", level="sample_name")
        """
        self._check_wavelength()
        return self._to_pandas(rt_grid, rt_atol)


class SpectrumSet(_TraceSet):
    """Spectra at one time from several runs.

    Behaves like a read-only list of Spectrum objects, and converts to a
    runs x wavelength matrix with ``to_numpy`` or ``to_pandas``. Rows are
    labelled by (run, sample_name).
    """

    _axis = "wl"
    _axis_label = "wl_nm"
    _unit = "nm"

    def savgol(self, window: float, *, polyorder: int = 2, deriv: int = 0) -> "SpectrumSet":
        """Smooth every spectrum; see ``Spectrum.savgol`` (window in nm)."""
        return self._apply("savgol", window, polyorder=polyorder, deriv=deriv)

    def to_numpy(self, *, wl_grid=None, wl_atol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        """Return (wl, X) with X of shape (n_runs, n_wl), in mAU.

        wl_grid works like ``rt_grid`` in ``ChromatogramSet.to_numpy``.
        """
        return self._to_numpy(wl_grid, wl_atol)

    def to_pandas(self, *, wl_grid=None, wl_atol: float = 1e-6) -> pd.DataFrame:
        """Return a DataFrame with one row per run and one column per wavelength (nm).

        Rows are indexed by (run, sample_name).
        """
        return self._to_pandas(wl_grid, wl_atol)


# ---------------------------------------------------------------- alignment

def same_axis(a: np.ndarray, b: np.ndarray, atol: float) -> bool:
    """True if two axes have the same length and values within atol."""
    return len(a) == len(b) and bool(np.allclose(a, b, rtol=0, atol=atol))


def interp_onto(grid: np.ndarray, axis: np.ndarray, signal: np.ndarray, unit: str = "") -> np.ndarray:
    """Linearly interpolate a 1-D signal, or a 2-D one along its first axis."""
    eps = 1e-9
    if grid[0] < axis[0] - eps or grid[-1] > axis[-1] + eps:
        raise ValueError(
            f"Grid ({grid[0]:g}-{grid[-1]:g} {unit}) reaches outside a run "
            f"({axis[0]:g}-{axis[-1]:g} {unit}); extrapolation is not allowed."
        )
    if signal.ndim == 1:
        return np.interp(grid, axis, signal)
    return np.column_stack(
        [np.interp(grid, axis, signal[:, j]) for j in range(signal.shape[1])]
    )


def align_traces(
    axes: list[np.ndarray],
    signals: list[np.ndarray],
    grid,
    atol: float,
    *,
    name: str,
    unit: str,
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Put signals with (possibly) different axes onto one axis and stack them.

    grid: None (axes must already match), "overlap", or an array.
    Returns (axis, stacked_signals).
    """
    if len(axes) == 0:
        raise ValueError("Nothing to combine: the set is empty.")

    if grid is None:
        ref = axes[0]
        for i in range(1, len(axes)):
            if not same_axis(axes[i], ref, atol):
                ax = axes[i]
                raise ValueError(
                    f"{labels[i]} has a different {name} axis ({ax[0]:g}-{ax[-1]:g} {unit}, "
                    f"{len(ax)} points) than {labels[0]} ({ref[0]:g}-{ref[-1]:g} {unit}, "
                    f"{len(ref)} points). Pass {name}_grid='overlap' or an array to "
                    f"interpolate onto a common grid."
                )
        return np.asarray(ref).copy(), np.stack(signals)

    if isinstance(grid, str):
        if grid != "overlap":
            raise ValueError(f"Unknown {name}_grid {grid!r}; use None, 'overlap', or an array.")
        start = max(ax[0] for ax in axes)
        end = min(ax[-1] for ax in axes)
        ref = axes[0]
        new_axis = ref[(ref >= start) & (ref <= end)].copy()
        if len(new_axis) < 2:
            raise ValueError(
                f"The runs share no common {name} range (latest start {start:g} {unit}, "
                f"earliest end {end:g} {unit})."
            )
    else:
        new_axis = np.asarray(grid, dtype=float)
        if new_axis.ndim != 1 or len(new_axis) < 2 or np.any(np.diff(new_axis) <= 0):
            raise ValueError(f"{name}_grid must be a 1-D, strictly increasing array.")

    return new_axis, np.stack(
        [interp_onto(new_axis, ax, s, unit) for ax, s in zip(axes, signals)]
    )


# ----------------------------------------------------------------- extraction

def extract_spectrum(
    run,
    time: float,
    time_tolerance: float = 0.5,
    *,
    wl_min: float | None = None,
    wl_max: float | None = None,
) -> Spectrum:
    """Return the spectrum of `run` at the time point nearest to `time` (min).

    Raises ValueError if that point is more than `time_tolerance` min away,
    or if the wavelength range contains no points. Used by
    DADRun.get_spectrum, which documents the options for users.
    """
    idx = abs(run.rt - time).argmin()
    chosen_rt = run.rt[idx]

    if abs(chosen_rt - time) > time_tolerance:
        raise ValueError(
            f"Nearest available time to {time} min is "
            f"{chosen_rt} min, more than {time_tolerance} min away."
        )

    start = 0 if wl_min is None else int(np.searchsorted(run.wl, wl_min, side="left"))
    stop = len(run.wl) if wl_max is None else int(np.searchsorted(run.wl, wl_max, side="right"))

    if start >= stop:
        raise ValueError(
            f"Wavelength range ({wl_min}, {wl_max}) nm contains no points; "
            f"the run covers {run.wl[0]} to {run.wl[-1]} nm."
        )

    return Spectrum(
        time=float(chosen_rt),
        wl=run.wl[start:stop].copy(),
        signal=run.signal[idx, start:stop].copy(),
        meta=run.meta,
        requested_time=time,
    )


def extract_chromatogram(
    run,
    wavelength: float,
    wl_tolerance: float = 1.0,
    *,
    rt_min: float | None = None,
    rt_max: float | None = None,
) -> Chromatogram:
    """Return the chromatogram of `run` at the wavelength nearest to `wavelength` (nm).

    Raises ValueError if that wavelength is more than `wl_tolerance` nm away,
    or if the time range contains no points. Used by DADRun.get_chromatogram,
    which documents the options for users.
    """
    idx = abs(run.wl - wavelength).argmin()
    chosen_wl = run.wl[idx]

    if abs(chosen_wl - wavelength) > wl_tolerance:
        raise ValueError(
            f"Nearest available wavelength to {wavelength} nm is "
            f"{chosen_wl} nm, more than {wl_tolerance} nm away."
        )

    start = 0 if rt_min is None else int(np.searchsorted(run.rt, rt_min, side="left"))
    stop = len(run.rt) if rt_max is None else int(np.searchsorted(run.rt, rt_max, side="right"))

    if start >= stop:
        raise ValueError(
            f"Time range ({rt_min}, {rt_max}) min contains no points; "
            f"the run covers {run.rt[0]} to {run.rt[-1]} min."
        )

    return Chromatogram(
        wavelength=float(chosen_wl),
        rt=run.rt[start:stop].copy(),
        signal=run.signal[start:stop, idx].copy(),
        meta=run.meta,
        requested_wavelength=wavelength,
    )
