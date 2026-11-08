from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .meta import DADMeta
from .preprocessing import asls_baseline, savgol_smooth


@dataclass(frozen=True)
class Spectrum:
    """A spectrum extracted from a DAD run: intensity vs wavelength at one time."""

    time: float          # time point used, in min
    wavelength: np.ndarray       # wavelength axis, in nm
    intensity: np.ndarray   # mAU
    meta: DADMeta
    requested_time: float | None = None   # time asked for, in min
    processing: tuple[str, ...] = ()      # steps applied, in order

    def __post_init__(self):
        if np.shape(self.intensity) != np.shape(self.wavelength):
            raise ValueError(
                f"Intensity shape {np.shape(self.intensity)} does not match "
                f"wavelength axis shape {np.shape(self.wavelength)}."
            )

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (wavelength, intensity) as copies."""
        return self.wavelength.copy(), self.intensity.copy()

    def to_pandas(self) -> pd.Series:
        """Return the intensity as a Series indexed by wavelength (nm)."""
        return pd.Series(
            self.intensity.copy(),
            index=pd.Index(self.wavelength.copy(), name="wavelength_nm"),
            name=self.meta.sample_name,
        )

    def savgol(self, window_length: int, *, polyorder: int = 2, deriv: int = 0) -> "Spectrum":
        """Return a new Spectrum smoothed with a Savitzky-Golay filter along wavelength.

        Parameters
        ----------
        window_length : int
            Window length in points; must be odd and larger than polyorder.
            With a 2 nm step, 5 points span 8 nm.
        polyorder : int, default 2
            Order of the fitted polynomial.
        deriv : int, default 0
            Derivative order; 1 or 2 give spectral derivatives (mAU/nm, mAU/nm²).

        Examples
        --------
        >>> smooth = spectrum.savgol(5)
        >>> first_derivative = spectrum.savgol(5, deriv=1)
        """
        intensity = savgol_smooth(
            self.intensity, self.wavelength, window_length, polyorder=polyorder, deriv=deriv, unit="nm"
        )
        step = f"savgol (window_length={window_length} points, polyorder={polyorder}, deriv={deriv})"
        return replace(self, intensity=intensity, processing=self.processing + (step,))


@dataclass(frozen=True)
class Chromatogram:
    """A chromatogram extracted from a DAD run: intensity vs time at one wavelength."""

    wavelength: float    # wavelength used, in nm
    time: np.ndarray       # time axis, in min
    intensity: np.ndarray   # mAU
    meta: DADMeta
    requested_wavelength: float | None = None   # wavelength asked for, in nm
    processing: tuple[str, ...] = ()            # steps applied, in order

    def __post_init__(self):
        if np.shape(self.intensity) != np.shape(self.time):
            raise ValueError(
                f"Intensity shape {np.shape(self.intensity)} does not match "
                f"time axis shape {np.shape(self.time)}."
            )

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (time, intensity) as copies."""
        return self.time.copy(), self.intensity.copy()

    def to_pandas(self) -> pd.Series:
        """Return the intensity as a Series indexed by time (min)."""
        return pd.Series(
            self.intensity.copy(),
            index=pd.Index(self.time.copy(), name="time_min"),
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
        baseline = asls_baseline(self.intensity, lam=lam, p=p, n_iter=n_iter)
        step = f"asls baseline estimate (lam={lam:g}, p={p:g})"
        return replace(self, intensity=baseline, processing=self.processing + (step,))

    def correct_baseline(
        self, *, lam: float = 1e6, p: float = 0.01, n_iter: int = 10
    ) -> "Chromatogram":
        """Return a new Chromatogram with the baseline removed (asymmetric least squares).

        Fits a smooth curve along the lower edge of the intensity, leaving out
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
        baseline = asls_baseline(self.intensity, lam=lam, p=p, n_iter=n_iter)
        step = f"asls baseline correction (lam={lam:g}, p={p:g})"
        return replace(self, intensity=self.intensity - baseline, processing=self.processing + (step,))

    def savgol(self, window_length: int, *, polyorder: int = 2, deriv: int = 0) -> "Chromatogram":
        """Return a new Chromatogram smoothed with a Savitzky-Golay filter along time.

        Parameters
        ----------
        window_length : int
            Window length in points; must be odd and larger than polyorder.
            Keep it well below the number of points across your narrowest
            peak, or peaks get flattened. At 2.5 Hz, 11 points span 4 s.
        polyorder : int, default 2
            Order of the fitted polynomial.
        deriv : int, default 0
            Derivative order (mAU/min for 1).

        Examples
        --------
        >>> smooth = chromatogram.savgol(11)
        """
        intensity = savgol_smooth(
            self.intensity, self.time, window_length, polyorder=polyorder, deriv=deriv, unit="min"
        )
        step = f"savgol (window_length={window_length} points, polyorder={polyorder}, deriv={deriv})"
        return replace(self, intensity=intensity, processing=self.processing + (step,))


class _TraceSet:
    """Shared behaviour of ChromatogramSet and SpectrumSet.

    Holds one trace per run, together with each run's position in the
    DADSequence it came from, so labels survive slicing.
    """

    _axis = ""        # "time" or "wavelength": attribute name of the x axis
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
            [t.intensity for t in self._traces],
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

    _axis = "time"
    _axis_label = "time_min"
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

    def savgol(self, window_length: int, *, polyorder: int = 2, deriv: int = 0) -> "ChromatogramSet":
        """Smooth every chromatogram; see ``Chromatogram.savgol`` (window in points)."""
        return self._apply("savgol", window_length, polyorder=polyorder, deriv=deriv)

    def to_numpy(self, *, time_grid=None, time_atol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        """Return (time, X) with X of shape (n_runs, n_time), in mAU.

        Parameters
        ----------
        time_grid : None, "overlap", or array, default None
            None requires every chromatogram to have the same time axis.
            "overlap" interpolates onto the first run's time points inside the
            range covered by every run. An array interpolates onto those
            times (min), which must lie inside every run.
        time_atol : float, default 1e-6
            Tolerance in minutes when checking that the time axes match.
        """
        self._check_wavelength()
        return self._to_numpy(time_grid, time_atol)

    def to_pandas(self, *, time_grid=None, time_atol: float = 1e-6) -> pd.DataFrame:
        """Return a DataFrame with one row per run and one column per time (min).

        Rows are indexed by (run, sample_name); see ``to_numpy`` for options.

        Examples
        --------
        >>> df = seq.get_chromatogram(314, time_min=5, time_max=40).to_pandas()
        >>> df.xs("QC", level="sample_name")
        """
        self._check_wavelength()
        return self._to_pandas(time_grid, time_atol)


class SpectrumSet(_TraceSet):
    """Spectra at one time from several runs.

    Behaves like a read-only list of Spectrum objects, and converts to a
    runs x wavelength matrix with ``to_numpy`` or ``to_pandas``. Rows are
    labelled by (run, sample_name).
    """

    _axis = "wavelength"
    _axis_label = "wavelength_nm"
    _unit = "nm"

    def savgol(self, window_length: int, *, polyorder: int = 2, deriv: int = 0) -> "SpectrumSet":
        """Smooth every spectrum; see ``Spectrum.savgol`` (window in points)."""
        return self._apply("savgol", window_length, polyorder=polyorder, deriv=deriv)

    def to_numpy(self, *, wavelength_grid=None, wavelength_atol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
        """Return (wavelength, X) with X of shape (n_runs, n_wavelength), in mAU.

        wavelength_grid works like ``time_grid`` in ``ChromatogramSet.to_numpy``.
        """
        return self._to_numpy(wavelength_grid, wavelength_atol)

    def to_pandas(self, *, wavelength_grid=None, wavelength_atol: float = 1e-6) -> pd.DataFrame:
        """Return a DataFrame with one row per run and one column per wavelength (nm).

        Rows are indexed by (run, sample_name).
        """
        return self._to_pandas(wavelength_grid, wavelength_atol)


# ---------------------------------------------------------------- alignment

def same_axis(a: np.ndarray, b: np.ndarray, atol: float) -> bool:
    """True if two axes have the same length and values within atol."""
    return len(a) == len(b) and bool(np.allclose(a, b, rtol=0, atol=atol))


def interp_onto(grid: np.ndarray, axis: np.ndarray, intensity: np.ndarray, unit: str = "") -> np.ndarray:
    """Linearly interpolate a 1-D intensity, or a 2-D one along its first axis."""
    eps = 1e-9
    if grid[0] < axis[0] - eps or grid[-1] > axis[-1] + eps:
        raise ValueError(
            f"Grid ({grid[0]:g}-{grid[-1]:g} {unit}) reaches outside a run "
            f"({axis[0]:g}-{axis[-1]:g} {unit}); extrapolation is not allowed."
        )
    if intensity.ndim == 1:
        return np.interp(grid, axis, intensity)
    return np.column_stack(
        [np.interp(grid, axis, intensity[:, j]) for j in range(intensity.shape[1])]
    )


def align_traces(
    axes: list[np.ndarray],
    intensities: list[np.ndarray],
    grid,
    atol: float,
    *,
    name: str,
    unit: str,
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Put intensities with (possibly) different axes onto one axis and stack them.

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
        return np.asarray(ref).copy(), np.stack(intensities)

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
        [interp_onto(new_axis, ax, s, unit) for ax, s in zip(axes, intensities)]
    )


# ----------------------------------------------------------------- extraction

def extract_spectrum(
    run,
    time: float,
    time_tolerance: float = 0.5,
    *,
    wavelength_min: float | None = None,
    wavelength_max: float | None = None,
) -> Spectrum:
    """Return the spectrum of `run` at the time point nearest to `time` (min).

    Raises ValueError if that point is more than `time_tolerance` min away,
    or if the wavelength range contains no points. Used by
    DADRun.get_spectrum, which documents the options for users.
    """
    idx = abs(run.time - time).argmin()
    chosen_time = run.time[idx]

    if abs(chosen_time - time) > time_tolerance:
        raise ValueError(
            f"Nearest available time to {time} min is "
            f"{chosen_time} min, more than {time_tolerance} min away."
        )

    start = 0 if wavelength_min is None else int(np.searchsorted(run.wavelength, wavelength_min, side="left"))
    stop = len(run.wavelength) if wavelength_max is None else int(np.searchsorted(run.wavelength, wavelength_max, side="right"))

    if start >= stop:
        raise ValueError(
            f"Wavelength range ({wavelength_min}, {wavelength_max}) nm contains no points; "
            f"the run covers {run.wavelength[0]} to {run.wavelength[-1]} nm."
        )

    return Spectrum(
        time=float(chosen_time),
        wavelength=run.wavelength[start:stop].copy(),
        intensity=run.intensity[idx, start:stop].copy(),
        meta=run.meta,
        requested_time=time,
    )


def extract_chromatogram(
    run,
    wavelength: float,
    wavelength_tolerance: float = 1.0,
    *,
    time_min: float | None = None,
    time_max: float | None = None,
) -> Chromatogram:
    """Return the chromatogram of `run` at the wavelength nearest to `wavelength` (nm).

    Raises ValueError if that wavelength is more than `wavelength_tolerance` nm away,
    or if the time range contains no points. Used by DADRun.get_chromatogram,
    which documents the options for users.
    """
    idx = abs(run.wavelength - wavelength).argmin()
    chosen_wavelength = run.wavelength[idx]

    if abs(chosen_wavelength - wavelength) > wavelength_tolerance:
        raise ValueError(
            f"Nearest available wavelength to {wavelength} nm is "
            f"{chosen_wavelength} nm, more than {wavelength_tolerance} nm away."
        )

    start = 0 if time_min is None else int(np.searchsorted(run.time, time_min, side="left"))
    stop = len(run.time) if time_max is None else int(np.searchsorted(run.time, time_max, side="right"))

    if start >= stop:
        raise ValueError(
            f"Time range ({time_min}, {time_max}) min contains no points; "
            f"the run covers {run.time[0]} to {run.time[-1]} min."
        )

    return Chromatogram(
        wavelength=float(chosen_wavelength),
        time=run.time[start:stop].copy(),
        intensity=run.intensity[start:stop, idx].copy(),
        meta=run.meta,
        requested_wavelength=wavelength,
    )
