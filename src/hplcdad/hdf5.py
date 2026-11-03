"""Save DADRuns as HDF5 files and read them back.

One file per run, laid out with HDF5 dimension scales so that xarray can open
it too (xarray.open_dataset(path, engine="h5netcdf")):

    /rt       retention time, float64, attribute units="min"
    /wl       wavelength,     float64, attribute units="nm"
    /signal   (rt, wl),       float64, attribute units="mAU", compressed
    root attributes: the DADMeta fields (None fields are left out), plus
        hplcdad_schema_version, created (UTC, ISO 8601) and, when the source
        file still exists, source_sha256.

The signal is stored as float64, the same as DADRun holds it, so a run read
back is identical to the one written.
"""
import hashlib
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .meta import DADMeta
from .raw import RawRun

SCHEMA_VERSION = 1
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
        rt = file.create_dataset("rt", data=run.rt)
        rt.attrs["units"] = "min"
        rt.attrs["long_name"] = "retention time"
        rt.make_scale("rt")

        wl = file.create_dataset("wl", data=run.wl)
        wl.attrs["units"] = "nm"
        wl.attrs["long_name"] = "wavelength"
        wl.make_scale("wl")

        signal = file.create_dataset(
            "signal",
            data=run.signal,
            chunks=(min(len(run.rt), 1024), len(run.wl)),
            compression="gzip",
            compression_opts=compression_level,
            shuffle=True,
        )
        signal.attrs["units"] = "mAU"
        signal.attrs["long_name"] = "absorbance"
        signal.dims[0].attach_scale(rt)
        signal.dims[1].attach_scale(wl)

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
        for name in ("rt", "wl", "signal"):
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
            rt=np.array(file["rt"]),
            wl=np.array(file["wl"]),
            signal=np.array(file["signal"]),
            meta=meta,
            time_unit=file["rt"].attrs.get("units", "min"),
            signal_unit=file["signal"].attrs.get("units", "mAU"),
            rt_as_rows=True,
        )