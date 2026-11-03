from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from typing import Any
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd

from .meta import DADMeta
from .readers import read_cmbx
from .run import DADRun
from .traces import ChromatogramSet, SpectrumSet, align_traces, same_axis

META_FIELDS = tuple(f.name for f in fields(DADMeta))


class DADSequence:
    """An ordered, read-only collection of DADRun objects.

    Indexing
    --------
    seq[0]                  -> DADRun (position)
    seq[-1]                 -> DADRun (last run)
    seq["Brew 3"]           -> DADRun (sample name; must be unique)
    seq[2:5]                -> DADSequence (slice)
    seq[[0, 3, 7]]          -> DADSequence (positions)
    seq[["Brew 1", "QC"]]   -> DADSequence (unique sample names)
    seq[mask]               -> DADSequence (boolean mask, e.g. from meta_table())

    Methods that change the selection or the data (filter, sort_by, crop, +)
    return a new DADSequence and leave this one untouched.
    """

    def __init__(self, runs: Iterable[DADRun] = ()):
        runs = tuple(runs)
        for i, run in enumerate(runs):
            if not isinstance(run, DADRun):
                raise TypeError(
                    f"Item {i} is a {type(run).__name__}, not a DADRun."
                )
        self._runs = runs

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_chromeleon(
        cls,
        source,
        *,
        pattern: str = "*.txt",
        sort_by: str | Callable[[DADRun], Any] | None = "injection_datetime",
        time_unit: str = "min",
        signal_unit: str = "mAU",
        max_metadata_lines: int = 100,
    ) -> "DADSequence":
        """Read several Chromeleon 3D field exports into one DADSequence.

        Parameters
        ----------
        source : str, Path, or iterable of str/Path
            A folder (every file matching ``pattern`` is read), a single
            file, or a list of files.
        pattern : str, default "*.txt"
            Glob pattern used when ``source`` is a folder.
        sort_by : str, callable, or None, default "injection_datetime"
            How to order the runs (see ``sort_by``). None keeps the order of
            the files (alphabetical for a folder).
        time_unit, signal_unit, max_metadata_lines
            Passed on to ``DADRun.from_chromeleon`` for every file.

        Raises
        ------
        FileNotFoundError
            If the folder contains no matching files or a path does not exist.
        ValueError
            If a file cannot be read; the message names the file.

        Examples
        --------
        >>> seq = DADSequence.from_chromeleon("data/2026-10-19")
        >>> seq = DADSequence.from_chromeleon("data", pattern="Brew*.txt")
        """
        runs = []
        for path in _resolve_paths(source, pattern):
            try:
                runs.append(
                    DADRun.from_chromeleon(
                        path,
                        time_unit=time_unit,
                        signal_unit=signal_unit,
                        max_metadata_lines=max_metadata_lines,
                    )
                )
            except ValueError as e:
                raise ValueError(f"Could not load {str(path)!r}: {e}") from e

        sequence = cls(runs)
        return sequence.sort_by(sort_by) if sort_by is not None else sequence

    @classmethod
    def from_cmbx(
        cls,
        source,
        *,
        pattern: str = "*.cmbx",
        sort_by: str | Callable[[DADRun], Any] | None = "injection_datetime",
    ) -> "DADSequence":
        """Read the 3D fields straight from Chromeleon backup archives (.cmbx).

        Every injection with a 3D field becomes a run; injections without
        one (e.g. a wash) are skipped. No export from Chromeleon is needed.

        Compared with ``from_chromeleon`` (text exports), the metadata are
        more limited. ``injection_datetime`` comes from the name Chromeleon
        gives the 3D field file, which is 0-1.1 s after the actual injection.
        ``injection_number`` is the injection's position in the sequence.
        ``injection_volume_ul``, ``vial_position`` and ``method`` are None.
        ``injection_id`` (e.g. "556.smp") is Chromeleon's own id and tells
        apart injections that share a name.

        Parameters
        ----------
        source : str, Path, or iterable of str/Path
            A folder (every file matching ``pattern`` is read), a single
            .cmbx file, or a list of files.
        pattern : str, default "*.cmbx"
            Glob pattern used when ``source`` is a folder.
        sort_by : str, callable, or None, default "injection_datetime"
            How to order the runs (see ``sort_by``). None keeps the order of
            the archives and, within each, the sequence order.

        Raises
        ------
        FileNotFoundError
            If the folder contains no matching files or a path does not exist.
        ValueError
            If a file is not a .cmbx archive or its 3D data cannot be read;
            the message names the file.

        Notes
        -----
        The .sfd format was worked out from Chromeleon 7.3.2 data recorded
        at 190-640 nm, 2 nm steps, 2.5 Hz, and checked against text exports
        of the same injections. Check one export against this reader after
        changing the DAD settings or updating Chromeleon.

        Examples
        --------
        >>> seq = DADSequence.from_cmbx("data/20261021 - Spike experiment.cmbx")
        >>> seq = DADSequence.from_cmbx("data/backups")
        """
        runs = []
        for path in _resolve_paths(source, pattern):
            try:
                runs.extend(DADRun.from_raw(raw) for raw in read_cmbx(path))
            except ValueError as e:
                raise ValueError(f"Could not load {str(path)!r}: {e}") from e

        sequence = cls(runs)
        return sequence.sort_by(sort_by) if sort_by is not None else sequence

    @classmethod
    def from_hdf5(
        cls,
        source,
        *,
        pattern: str = "*.h5",
        sort_by: str | Callable[[DADRun], Any] | None = "injection_datetime",
    ) -> "DADSequence":
        """Read runs saved with ``to_hdf5``.

        Parameters
        ----------
        source : str, Path, or iterable of str/Path
            A folder (every file matching ``pattern`` is read), a single
            file, or a list of files.
        pattern : str, default "*.h5"
            Glob pattern used when ``source`` is a folder.
        sort_by : str, callable, or None, default "injection_datetime"
            How to order the runs (see ``sort_by``).

        Examples
        --------
        >>> seq = DADSequence.from_hdf5("converted")
        """
        runs = []
        for path in _resolve_paths(source, pattern):
            try:
                runs.append(DADRun.from_hdf5(path))
            except ValueError as e:
                raise ValueError(f"Could not load {str(path)!r}: {e}") from e

        sequence = cls(runs)
        return sequence.sort_by(sort_by) if sort_by is not None else sequence

    def to_hdf5(
        self,
        folder,
        *,
        overwrite: bool = False,
        compression_level: int = 4,
    ) -> list[Path]:
        """Save every run as its own HDF5 file in ``folder`` and return the paths.

        Files are named after the sample (``BLANK-01.h5``). When names repeat,
        the Chromeleon injection id is added (``OPTI_01 (836.smp).h5``), or
        the injection number if there is no id. Characters that Windows does
        not allow in file names are replaced with "_". The folder is created
        if needed.

        Raises
        ------
        FileExistsError
            If a file already exists and ``overwrite`` is False. Nothing is
            written in that case.

        Examples
        --------
        >>> seq = DADSequence.from_cmbx("data/20261021 - Spike experiment.cmbx")
        >>> seq.to_hdf5("converted/20261021 - Spike experiment")
        """
        paths = self._file_paths(folder, ".h5", overwrite)
        for run, path in zip(self._runs, paths):
            run.to_hdf5(path, overwrite=overwrite, compression_level=compression_level)
        return paths

    def to_chromeleon(self, folder, *, overwrite: bool = False) -> list[Path]:
        """Save every run as a Chromeleon 3D field text export in ``folder``.

        Files are named like ``to_hdf5`` names them, with ``.txt``
        (``BLANK-01.txt``; ``OPTI_01 (836.smp).txt`` when names repeat).
        See ``DADRun.to_chromeleon`` for what is written.

        Raises
        ------
        FileExistsError
            If a file already exists and ``overwrite`` is False. Nothing is
            written in that case.

        Examples
        --------
        >>> seq = DADSequence.from_cmbx("data/20261021 - Spike experiment.cmbx")
        >>> seq.to_chromeleon("exports/20261021 - Spike experiment")
        """
        paths = self._file_paths(folder, ".txt", overwrite)
        for run, path in zip(self._runs, paths):
            run.to_chromeleon(path, overwrite=overwrite)
        return paths

    def _file_paths(self, folder, suffix: str, overwrite: bool) -> list[Path]:
        """One file path per run in ``folder``, named after the sample.

        Repeated names get the injection id (or number) added. Raises before
        anything is written if two runs would share a name, or if a file
        exists and ``overwrite`` is False. Creates the folder.
        """
        folder = Path(folder)
        names = [run.meta.sample_name for run in self._runs]
        paths = []
        for i, run in enumerate(self._runs):
            stem = run.meta.sample_name
            if names.count(stem) > 1:
                tag = run.meta.injection_id or f"#{run.meta.injection_number or i + 1}"
                stem = f"{stem} ({tag})"
            paths.append(folder / f"{_safe_file_name(stem)}{suffix}")

        if len(set(paths)) != len(paths):
            raise ValueError("Two runs would be saved under the same file name.")
        if not overwrite:
            existing = [str(p) for p in paths if p.exists()]
            if existing:
                raise FileExistsError(
                    f"{len(existing)} file(s) already exist, e.g. {existing[0]!r}; "
                    f"pass overwrite=True to replace them."
                )

        folder.mkdir(parents=True, exist_ok=True)
        return paths

    # ------------------------------------------------------- container protocol

    def __len__(self) -> int:
        return len(self._runs)

    def __iter__(self) -> Iterator[DADRun]:
        return iter(self._runs)

    def __getitem__(self, key):
        if isinstance(key, (bool, np.bool_)):
            raise TypeError("Index with an int, str, slice, list, or boolean mask.")
        if isinstance(key, (int, np.integer)):
            return self._runs[key]
        if isinstance(key, str):
            return self.get(key)
        if isinstance(key, slice):
            return DADSequence(self._runs[key])

        key = np.asarray(key)
        if key.ndim != 1:
            raise TypeError(f"Index arrays must be 1-dimensional, got shape {key.shape}.")
        if key.dtype == bool:
            if len(key) != len(self):
                raise IndexError(
                    f"Boolean mask has length {len(key)}, but the sequence "
                    f"has {len(self)} runs."
                )
            return DADSequence(run for run, keep in zip(self._runs, key) if keep)
        if np.issubdtype(key.dtype, np.integer):
            return DADSequence(self._runs[i] for i in key)
        if key.dtype.kind in "UO" and all(isinstance(k, str) for k in key):
            return DADSequence(self.get(name) for name in key)
        if key.size == 0:
            return DADSequence()
        raise TypeError(
            f"Cannot index a DADSequence with {type(key).__name__} of dtype {key.dtype}."
        )

    def __add__(self, other) -> "DADSequence":
        if isinstance(other, DADSequence):
            return DADSequence(self._runs + other._runs)
        if isinstance(other, DADRun):
            return DADSequence(self._runs + (other,))
        return NotImplemented

    def __repr__(self) -> str:
        lines = [f"DADSequence({len(self)} run{'s' if len(self) != 1 else ''})"]
        shown = self._runs if len(self) <= 12 else self._runs[:10]
        width = max((len(r.meta.sample_name) for r in shown), default=0)
        for i, run in enumerate(shown):
            when = run.meta.injection_datetime.strftime("%Y-%m-%d %H:%M")
            lines.append(f"  [{i}] {run.meta.sample_name:<{width}}  {when}")
        if len(shown) < len(self):
            lines.append(f"  ... {len(self) - len(shown)} more")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        """Show the metadata table in Jupyter."""
        return self.meta_table().to_html()

    # ------------------------------------------------------------- metadata

    @property
    def runs(self) -> tuple[DADRun, ...]:
        """The runs as a tuple."""
        return self._runs

    @property
    def names(self) -> list[str]:
        """Sample names, in sequence order."""
        return [run.meta.sample_name for run in self._runs]

    @property
    def meta(self) -> list[DADMeta]:
        """Metadata of every run, in sequence order."""
        return [run.meta for run in self._runs]

    def meta_table(self) -> pd.DataFrame:
        """Return one row per run with its metadata and axis ranges.

        The index is the position in the sequence, so a boolean column test
        on this table can be used directly as a mask:

        >>> table = seq.meta_table()
        >>> qcs = seq[table["sample_name"].str.startswith("QC")]
        """
        columns = list(META_FIELDS) + [
            "n_time", "n_wl", "rt_start", "rt_end", "wl_start", "wl_end",
        ]
        rows = []
        for run in self._runs:
            row = asdict(run.meta)
            row.update(
                n_time=len(run.rt),
                n_wl=len(run.wl),
                rt_start=float(run.rt[0]),
                rt_end=float(run.rt[-1]),
                wl_start=float(run.wl[0]),
                wl_end=float(run.wl[-1]),
            )
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    # ------------------------------------------------------------ selection

    def get(self, sample_name: str) -> DADRun:
        """Return the single run with this sample name.

        Raises
        ------
        KeyError
            If no run, or more than one run, has this name. For replicate
            injections with the same name, use ``filter(sample_name=...)``.
        """
        matches = [run for run in self._runs if run.meta.sample_name == sample_name]
        if not matches:
            raise KeyError(
                f"No run named {sample_name!r}. Available names: {sorted(set(self.names))}"
            )
        if len(matches) > 1:
            raise KeyError(
                f"{len(matches)} runs are named {sample_name!r}. "
                f"Use seq.filter(sample_name={sample_name!r}) to get all of them."
            )
        return matches[0]

    def filter(
        self,
        predicate: Callable[[DADRun], bool] | None = None,
        /,
        *,
        match: str | None = None,
        **meta_fields,
    ) -> "DADSequence":
        """Return the runs that meet every given condition.

        Parameters
        ----------
        predicate : callable, optional
            Function taking a DADRun and returning True to keep it.
        match : str, optional
            Regular expression searched for in the sample name.
        **meta_fields
            Exact matches on DADMeta fields, e.g. ``method="Bitter_acids"``.

        Examples
        --------
        >>> seq.filter(match=r"^Brew")
        >>> seq.filter(sample_name="QC")
        >>> seq.filter(lambda run: run.rt[-1] > 30)
        """
        for name in meta_fields:
            _check_meta_field(name)
        pattern = re.compile(match) if match is not None else None

        def keep(run: DADRun) -> bool:
            if pattern is not None and not pattern.search(run.meta.sample_name):
                return False
            for name, value in meta_fields.items():
                if getattr(run.meta, name) != value:
                    return False
            return predicate is None or bool(predicate(run))

        return DADSequence(run for run in self._runs if keep(run))

    def sort_by(
        self,
        key: str | Callable[[DADRun], Any] = "injection_datetime",
        *,
        reverse: bool = False,
    ) -> "DADSequence":
        """Return the runs ordered by a DADMeta field or a key function.

        Runs whose key is None go last, whatever ``reverse`` is.

        Examples
        --------
        >>> seq.sort_by("sample_name")
        >>> seq.sort_by(lambda run: run.meta.injection_number or 0)
        """
        if callable(key):
            key_function = key
        else:
            _check_meta_field(key)
            key_function = lambda run: getattr(run.meta, key)  # noqa: E731

        present = [run for run in self._runs if key_function(run) is not None]
        missing = [run for run in self._runs if key_function(run) is None]
        return DADSequence(sorted(present, key=key_function, reverse=reverse) + missing)

    # ------------------------------------------------------- per-run operations

    def crop(
        self,
        *,
        rt_min: float | None = None,
        rt_max: float | None = None,
        wl_min: float | None = None,
        wl_max: float | None = None,
    ) -> "DADSequence":
        """Crop every run (see ``DADRun.crop``) and return a new DADSequence."""
        return DADSequence(
            run.crop(rt_min=rt_min, rt_max=rt_max, wl_min=wl_min, wl_max=wl_max)
            for run in self._runs
        )

    def get_chromatogram(
        self,
        wavelength: float,
        wl_tolerance: float = 1.0,
        *,
        rt_min: float | None = None,
        rt_max: float | None = None,
    ) -> ChromatogramSet:
        """Return the chromatogram of every run (see ``DADRun.get_chromatogram``).

        The result behaves like a list of Chromatogram objects and converts
        to a runs x time matrix with ``to_numpy()`` or ``to_pandas()``, with
        rows labelled by (run, sample_name).

        Examples
        --------
        >>> rt, X = seq.get_chromatogram(314, rt_min=5, rt_max=40).to_numpy()
        >>> df = seq.get_chromatogram(314).to_pandas(rt_grid="overlap")
        """
        return ChromatogramSet(
            (run.get_chromatogram(wavelength, wl_tolerance, rt_min=rt_min, rt_max=rt_max)
             for run in self._runs),
            run_index=range(len(self)),
        )

    def get_spectrum(
        self,
        time: float,
        time_tolerance: float = 0.5,
        *,
        wl_min: float | None = None,
        wl_max: float | None = None,
    ) -> SpectrumSet:
        """Return the spectrum of every run (see ``DADRun.get_spectrum``).

        The result behaves like a list of Spectrum objects and converts to a
        runs x wavelength matrix with ``to_numpy()`` or ``to_pandas()``, with
        rows labelled by (run, sample_name).

        Examples
        --------
        >>> wl, X = seq.get_spectrum(12.4, wl_min=220, wl_max=400).to_numpy()
        """
        return SpectrumSet(
            (run.get_spectrum(time, time_tolerance, wl_min=wl_min, wl_max=wl_max)
             for run in self._runs),
            run_index=range(len(self)),
        )

    # ------------------------------------------------- arrays for chemometrics

    def stack(
        self,
        *,
        rt_grid=None,
        rt_atol: float = 1e-6,
        wl_atol: float = 1e-6,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return all runs as one runs x time x wavelength array.

        Useful for PARAFAC/PARAFAC2 or for unfolding to a 2-D matrix.
        All runs must share the same wavelength axis. Crop first
        (``seq.crop(...)``) to keep the array to a manageable size.

        Parameters
        ----------
        rt_grid : None, "overlap", or array, default None
            None requires all runs to share the same time axis.
            "overlap" interpolates onto the first run's time points inside
            the time range covered by every run. An array interpolates onto
            those times (min), which must lie inside every run.
        rt_atol, wl_atol : float
            Tolerances (min, nm) when checking that axes are the same.

        Returns
        -------
        rt : ndarray, shape (n_time,)
        wl : ndarray, shape (n_wl,)
        X : ndarray, shape (n_runs, n_time, n_wl), in mAU

        Examples
        --------
        >>> rt, wl, X = seq.crop(rt_min=5, rt_max=40, wl_min=220, wl_max=400).stack()
        >>> X_unfolded = X.reshape(len(X), -1)   # runs x (time*wavelength)
        """
        self._require_runs()
        wl = self._common_wl(wl_atol)
        labels = [f"Run {i} ({name!r})" for i, name in enumerate(self.names)]
        rt, X = align_traces(
            [run.rt for run in self._runs],
            [run.signal for run in self._runs],
            rt_grid,
            rt_atol,
            name="rt",
            unit="min",
            labels=labels,
        )
        return rt, wl.copy(), X

    def has_common_axes(self, rt_atol: float = 1e-6, wl_atol: float = 1e-6) -> bool:
        """True if every run has the same time and wavelength axes."""
        if len(self) == 0:
            return True
        ref = self._runs[0]
        return all(
            same_axis(run.rt, ref.rt, rt_atol) and same_axis(run.wl, ref.wl, wl_atol)
            for run in self._runs[1:]
        )

    # --------------------------------------------------------------- helpers

    def _require_runs(self):
        if len(self) == 0:
            raise ValueError("The sequence is empty.")

    def _common_wl(self, atol: float) -> np.ndarray:
        ref = self._runs[0].wl
        for i, run in enumerate(self._runs[1:], start=1):
            if not same_axis(run.wl, ref, atol):
                raise ValueError(
                    f"Run {i} ({run.meta.sample_name!r}) has a different wavelength "
                    f"axis ({run.wl[0]:g}-{run.wl[-1]:g} nm, {len(run.wl)} points) "
                    f"than run 0 ({ref[0]:g}-{ref[-1]:g} nm, {len(ref)} points). "
                    f"Crop to a shared wavelength range first."
                )
        return ref


def _check_meta_field(name: str):
    if name not in META_FIELDS:
        raise ValueError(f"Unknown metadata field {name!r}. Known fields: {list(META_FIELDS)}.")


def _safe_file_name(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .") or "run"


def _resolve_paths(source, pattern: str) -> list[Path]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            paths = sorted(path.glob(pattern))
            if not paths:
                raise FileNotFoundError(f"No files matching {pattern!r} in {str(path)!r}.")
            return paths
        if path.is_file():
            return [path]
        raise FileNotFoundError(f"{str(path)!r} is neither a file nor a folder.")

    paths = [Path(p) for p in source]
    if not paths:
        raise ValueError("No paths given.")
    return paths