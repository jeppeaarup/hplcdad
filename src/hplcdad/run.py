import numpy as np
from .meta import DADMeta
from .processing import normalise
from .raw import RawRun
from .readers import read_chromeleon
from .hdf5 import read_hdf5, write_hdf5
from .writers import write_chromeleon
from .traces import Chromatogram, Spectrum, extract_spectrum, extract_chromatogram

class DADRun:
    """One DAD run: intensity is time x wavelength (time in min, wavelength in nm, mAU)."""

    def __init__(self, time: np.ndarray, wavelength: np.ndarray, intensity: np.ndarray, meta: DADMeta):
        time = np.asarray(time)
        wavelength = np.asarray(wavelength)
        intensity = np.asarray(intensity)

        if intensity.shape != (len(time), len(wavelength)):
            raise ValueError(
                f"Intensity shape {intensity.shape} does not match the axes: "
                f"expected ({len(time)}, {len(wavelength)}) = (time, wavelength)."
            )

        self.time = time
        self.wavelength = wavelength
        self.intensity = intensity
        self.meta = meta

    def __repr__(self):
        return (
            f"DADRun(sample={self.meta.sample_name!r}, "
            f"{self.intensity.shape[0]} x {self.intensity.shape[1]}, "
            f"{self.time[0]:g}-{self.time[-1]:g} min, "
            f"{self.wavelength[0]:g}-{self.wavelength[-1]:g} nm)"
        )

    @classmethod
    def from_raw(cls, raw: RawRun):
        """Build a DADRun from a RawRun, cleaning it first."""
        return cls(*normalise(raw))

    @classmethod
    def from_chromeleon(
        cls,
        path,
        *,
        time_unit: str = "min",
        intensity_unit: str = "mAU",
        max_metadata_lines: int = 100,
    ) -> "DADRun":
        """Read a Chromeleon 3D field export and return a DADRun.

        Reads the tab-separated ASCII "3DFIELD" spectral field export from
        Chromeleon (developed against exports from Chromeleon 7.2.7). The
        result always has time in minutes, wavelength in nm, and the intensity
        in mAU as a time x wavelength matrix, whatever units the file used.
        Sample and injection details (sample name, injection date and time,
        injection number, volume, vial position, method) are available in
        the ``meta`` attribute.

        Parameters
        ----------
        path : str or Path
            The exported text file.
        time_unit : str, default "min"
            Unit of the time column in the file ("min", "s", or "h").
        intensity_unit : str, default "mAU"
            Unit of the intensity values in the file ("mAU" or "AU").
        max_metadata_lines : int, default 100
            How many lines to search for the "Raw Data:" marker before
            concluding that the file is not a Chromeleon export.

        Returns
        -------
        DADRun

        Raises
        ------
        ValueError
            If the file is not a valid Chromeleon 3D field export (missing
            marker, header, data rows, or required metadata), if a unit
            label is not recognized, or if the cleaned data are
            inconsistent (for example a time axis that is not increasing).

        Examples
        --------
        >>> run = DADRun.from_chromeleon("data/sample.txt")
        >>> run = DADRun.from_chromeleon("data/sample.txt", intensity_unit="AU")
        """
        return cls.from_raw(
            read_chromeleon(
                path,
                time_unit=time_unit,
                intensity_unit=intensity_unit,
                max_metadata_lines=max_metadata_lines,
            )
        )

    @classmethod
    def from_hdf5(cls, path) -> "DADRun":
        """Read a run saved with ``to_hdf5``.

        Raises
        ------
        ValueError
            If the file was not written by hplcdad, uses a newer schema
            version, or lacks data or required metadata.
        ImportError
            If h5py is not installed.

        Examples
        --------
        >>> run = DADRun.from_hdf5("converted/BLANK-01.h5")
        """
        return cls.from_raw(read_hdf5(path))

    def to_hdf5(self, path, *, overwrite: bool = False, compression_level: int = 4):
        """Save the run as one HDF5 file and return its path.

        The file holds the time and wavelength axes, the intensity (float64,
        mAU, gzip-compressed) and every metadata field, so ``from_hdf5``
        gives back an identical run. When the source file still exists, its
        SHA-256 checksum is stored too, to trace the file to its raw data.
        The layout uses HDF5 dimension scales, so xarray can open it with
        ``xarray.open_dataset(path, engine="h5netcdf")``.

        Parameters
        ----------
        path : str or Path
            Where to write; ``.h5`` is the usual extension.
        overwrite : bool, default False
            Replace an existing file instead of raising FileExistsError.
        compression_level : int, default 4
            gzip level, 0 (none) to 9 (smallest, slowest).

        Examples
        --------
        >>> run.to_hdf5("converted/BLANK-01.h5")
        """
        return write_hdf5(self, path, overwrite=overwrite, compression_level=compression_level)

    def to_chromeleon(self, path, *, overwrite: bool = False, timezone: str = "Europe/Copenhagen"):
        """Save the run as a Chromeleon 3D field text export and return its path.

        Writes the layout Chromeleon 7.3.2 uses for its tab-separated 3D
        field export, so software that reads Chromeleon exports can open it,
        and ``from_chromeleon`` reads it back.

        Notes
        -----
        - Intensity values are written with 6 decimals, like Chromeleon does,
          so they are rounded to 1e-6 mAU.
        - Metadata fields that are None are left empty. The URL line is filled in only when
          the run has an ``injection_id``, with "hplcdad" as host, so it does
          not point to a real data vault.

        Parameters
        ----------
        path : str or Path
            Where to write; ``.txt`` is the usual extension.
        overwrite : bool, default False
            Replace an existing file instead of raising FileExistsError.
        timezone : str, default "Europe/Copenhagen"
            Time zone of the instrument, as an IANA name. The run's injection
            time is local time without a zone, so this sets the UTC offset
            written after it (e.g. "+02:00" in summer, "+01:00" in winter).

        Examples
        --------
        >>> run.to_chromeleon("exports/BLANK-01.txt")
        """
        return write_chromeleon(self, path, overwrite=overwrite, timezone=timezone)


    def get_chromatogram(
        self,
        wavelength: float,
        wavelength_tolerance: float = 1.0,
        *,
        time_min: float | None = None,
        time_max: float | None = None,
    ) -> Chromatogram:
        """Return the chromatogram at a given wavelength, optionally limited to a time range.

        The run is sampled at discrete wavelengths, so the chromatogram is
        taken at the wavelength nearest to the one requested. The returned
        object records both the wavelength you asked for and the one
        actually used.

        Parameters
        ----------
        wavelength : float
            Wavelength in nm.
        wavelength_tolerance : float, default 1.0
            Largest accepted distance, in nm, between the requested wavelength
            and the nearest wavelength in the run.
        time_min, time_max : float, optional
            Lower and upper time limit in minutes, both included. Leave out
            either one (or both) for no limit on that side.

        Returns
        -------
        Chromatogram
            With the attributes ``wavelength`` (wavelength used),
            ``requested_wavelength``, ``time`` (min), ``intensity`` (mAU), and
            ``meta``. The arrays are copies, so changing them does not
            change the run.

        Raises
        ------
        ValueError
            If the nearest wavelength is farther away than ``wavelength_tolerance``,
            or if the time range contains no points.

        Examples
        --------
        >>> chromatogram = run.get_chromatogram(254)
        >>> chromatogram = run.get_chromatogram(254, time_min=1.0, time_max=15.0)
        """
        return extract_chromatogram(
            self,
            wavelength,
            wavelength_tolerance,
            time_min=time_min,
            time_max=time_max,
        )

    
    def get_spectrum(
        self,
        time: float,
        time_tolerance: float = 0.5,
        *,
        wavelength_min: float | None = None,
        wavelength_max: float | None = None,
    ) -> Spectrum:
        """Return the spectrum at a given time, optionally limited to a wavelength range.

        The run is sampled at discrete time points, so the spectrum is taken
        at the point nearest to the requested time. The returned object
        records both the time you asked for and the time actually used.

        Parameters
        ----------
        time : float
            Time in minutes.
        time_tolerance : float, default 0.5
            Largest accepted distance, in minutes, between the requested time
            and the nearest time point in the run.
        wavelength_min, wavelength_max : float, optional
            Lower and upper wavelength limit in nm, both included. Leave out
            either one (or both) for no limit on that side.

        Returns
        -------
        Spectrum
            With the attributes ``time`` (time used), ``requested_time``,
            ``wavelength`` (nm), ``intensity`` (mAU), and ``meta``. The arrays are
            copies, so changing them does not change the run.

        Raises
        ------
        ValueError
            If the nearest time point is farther away than ``time_tolerance``,
            or if the wavelength range contains no points.

        Examples
        --------
        >>> spectrum = run.get_spectrum(2.0)
        >>> spectrum = run.get_spectrum(2.0, wavelength_min=220, wavelength_max=400)
        """
        return extract_spectrum(
            self,
            time,
            time_tolerance,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
        )

    # Short names for the two getters: run.chromatogram(254), run.spectrum(2.0).
    chromatogram = get_chromatogram
    spectrum = get_spectrum

    def crop(
    self,
    *,
    time_min: float | None = None,
    time_max: float | None = None,
    wavelength_min: float | None = None,
    wavelength_max: float | None = None,
) -> "DADRun":
        """Return a new DADRun limited to a time and/or wavelength range.

        Limits are inclusive and in the run's units (min and nm). Leave out
        a limit for no restriction on that side. The result holds copies of
        the data, so changing it does not change this run.

        Raises
        ------
        ValueError
            If either range contains no points.

        Examples
        --------
        >>> window = run.crop(time_min=5.0, time_max=20.0)
        >>> window = run.crop(time_min=5.0, time_max=20.0, wavelength_min=220, wavelength_max=400)
        """
        t_start = 0 if time_min is None else int(np.searchsorted(self.time, time_min, side="left"))
        t_stop = len(self.time) if time_max is None else int(np.searchsorted(self.time, time_max, side="right"))
        w_start = 0 if wavelength_min is None else int(np.searchsorted(self.wavelength, wavelength_min, side="left"))
        w_stop = len(self.wavelength) if wavelength_max is None else int(np.searchsorted(self.wavelength, wavelength_max, side="right"))

        if t_start >= t_stop:
            raise ValueError(
                f"Time range ({time_min}, {time_max}) min contains no points; "
                f"the run covers {self.time[0]:g} to {self.time[-1]:g} min."
            )
        if w_start >= w_stop:
            raise ValueError(
                f"Wavelength range ({wavelength_min}, {wavelength_max}) nm contains no points; "
                f"the run covers {self.wavelength[0]:g} to {self.wavelength[-1]:g} nm."
            )

        return DADRun(
            self.time[t_start:t_stop].copy(),
            self.wavelength[w_start:w_stop].copy(),
            self.intensity[t_start:t_stop, w_start:w_stop].copy(),
            self.meta,
        )