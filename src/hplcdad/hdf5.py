"""Save DADRuns as HDF5 files and read them back.

One file per run, laid out with HDF5 dimension scales so that xarray can open
it too (xarray.open_dataset(path, engine="h5netcdf")):

    /time        retention time, float64, attribute units="min"
    /wavelength  wavelength,     float64, attribute units="nm"
    /intensity   (time, wavelength), float64, attribute units="mAU", compressed
    root attributes: the DADMeta fields (None fields are left out), plus
        hplcdad_schema_version, created (UTC, ISO 8601) and, when the source
        file still exists, source_sha256.

The intensity is stored as float64, the same as DADRun holds it, so a run read
back is identical to the one written.
"""
import hashlib
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .meta import DADMeta
from .raw import RawRun

SCHEMA_VERSION = 3   # 2: rt -> time, signal -> intensity; 3: wl -> wavelength. Older files are still read
META_TYPES = {f.name: f.type for f in fields(DADMeta)}
INT_FIELDS = {"injection_number"}
FLOAT_FIELDS = {"injection_volume_ul"}
DATETIME_FIELDS = {"injection_datetime"}


def _h5py():
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "Reading and writing HDF5 files needs the h5py package "
            "(pip install h5py or conda install h5py)."
        ) from None
    return h5py


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_hdf5(run, path, *, overwrite=False, compression_level=4):
    """Write a DADRun to one HDF5 file. Internal; users call DADRun.to_hdf5."""
    h5py = _h5py()
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{str(path)!r} already exists; pass overwrite=True to replace it.")

    with h5py.File(path, "w") as file:
        time = file.create_dataset("time", data=run.time)
        time.attrs["units"] = "min"
        time.attrs["long_name"] = "retention time"
        time.make_scale("time")

        wavelength = file.create_dataset("wavelength", data=run.wavelength)
        wavelength.attrs["units"] = "nm"
        wavelength.attrs["long_name"] = "wavelength"
        wavelength.make_scale("wavelength")

        intensity = file.create_dataset(
            "intensity",
            data=run.intensity,
            chunks=(min(len(run.time), 1024), len(run.wavelength)),
            compression="gzip",
            compression_opts=compression_level,
            shuffle=True,
        )
        intensity.attrs["units"] = "mAU"
        intensity.attrs["long_name"] = "absorbance"
        intensity.dims[0].attach_scale(time)
        intensity.dims[1].attach_scale(wavelength)

        attrs = file.attrs
        attrs["hplcdad_schema_version"] = SCHEMA_VERSION
        attrs["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for name in META_TYPES:
            value = getattr(run.meta, name)
            if value is None:
                continue
            attrs[name] = value.isoformat() if name in DATETIME_FIELDS else value

        source = run.meta.source_path
        if source and Path(source).is_file():
            attrs["source_sha256"] = _sha256(source)

    return path


def read_hdf5(path):
    """Read an HDF5 file written by write_hdf5 into a RawRun. Internal; users call DADRun.from_hdf5."""
    h5py = _h5py()
    with h5py.File(path, "r") as file:
        version = file.attrs.get("hplcdad_schema_version")
        if version is None:
            raise ValueError(f"{str(path)!r} was not written by hplcdad (no hplcdad_schema_version).")
        if int(version) > SCHEMA_VERSION:
            raise ValueError(
                f"{str(path)!r} uses schema version {version}, but this version of "
                f"hplcdad only reads up to {SCHEMA_VERSION}. Update hplcdad."
            )
        # Older schemas used shorter dataset names.
        version = int(version)
        time_name = "rt" if version == 1 else "time"
        intensity_name = "signal" if version == 1 else "intensity"
        wavelength_name = "wl" if version < 3 else "wavelength"
        for name in (time_name, wavelength_name, intensity_name):
            if name not in file:
                raise ValueError(f"{str(path)!r} has no {name!r} dataset.")

        values = {}
        for name in META_TYPES:
            if name not in file.attrs:
                continue
            value = file.attrs[name]
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            if name in DATETIME_FIELDS:
                value = datetime.fromisoformat(value)
            elif name in INT_FIELDS:
                value = int(value)
            elif name in FLOAT_FIELDS:
                value = float(value)
            else:
                value = str(value)
            values[name] = value

        try:
            meta = DADMeta(**values)
        except TypeError as e:
            raise ValueError(f"{str(path)!r} is missing required metadata: {e}") from None

        return RawRun(
            time=np.array(file[time_name]),
            wavelength=np.array(file[wavelength_name]),
            intensity=np.array(file[intensity_name]),
            meta=meta,
            time_unit=file[time_name].attrs.get("units", "min"),
            intensity_unit=file[intensity_name].attrs.get("units", "mAU"),
            time_as_rows=True,
        )