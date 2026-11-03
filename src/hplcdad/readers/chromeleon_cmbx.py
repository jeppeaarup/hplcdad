"""Reader for Chromeleon backup archives (.cmbx)."""
import re
import struct
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

from ..meta import DADMeta
from ..raw import RawRun
from .chromeleon_txt import _injection_id_from_url

# --------------------------------------------------------------------------
# Chromeleon backup archives (.cmbx)
#
# A .cmbx file is a ZIP archive. header.xml lists every injection and the
# file holding each of its signals; the 3D DAD field is a ".sfd" file.
#
# An .sfd file holds one block per spectrum:
#   "RawSpHdr" | 8-byte length (38) | 38-byte header | float64 start wavelength | data
# The 38-byte header has the retention time (float64, min) at byte 3.
# The data are unsigned LEB128 varints. Each value has its sign in bit 6 and
# its magnitude in the other bits (see _sfd_signed). The first four values are
# settings: wavelength step x 10, an unknown value (10), and the signal scale
# as a fraction (1000 mAU / 16777215 counts). Then follows one value per
# wavelength: the first is the count at the start wavelength, the rest are
# differences from the previous wavelength. A "-0" (64) ends the spectrum;
# the few values after it are not decoded.
#
# Worked out from Chromeleon 7.3.2 data (190-640 nm, 2 nm, 2.5 Hz) and checked
# against 3D field text exports of the same injections: identical to within
# the exports' 6-decimal rounding.
# --------------------------------------------------------------------------

CMBX_INJECTION_TYPE = "Dionex.Chromeleon.Data.Injection"
SFD_BLOCK_MARKER = b"RawSpHdr"
SFD_HEADER_SIZE = 62  # marker (8) + length (8) + header (38) + start wavelength (8)
SFD_END_OF_SPECTRUM = 64


def _sfd_varints(data):
    values, current, shift = [], 0, 0
    for byte in data:
        current |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            values.append(current)
            current, shift = 0, 0
    return values


def _sfd_signed(value):
    """Bit 6 holds the sign; the bits below and above it hold the magnitude."""
    magnitude = ((value >> 7) << 6) | (value & 63)
    return -magnitude if value & 64 else magnitude


def _read_sfd(data, name):
    """Decode an .sfd 3D field into (retention time, wavelength, signal in mAU)."""
    starts = []
    position = data.find(SFD_BLOCK_MARKER)
    while position >= 0:
        starts.append(position)
        position = data.find(SFD_BLOCK_MARKER, position + 1)
    if not starts:
        raise ValueError(f"{name!r} contains no spectra (no {SFD_BLOCK_MARKER!r} blocks).")
    starts.append(len(data))

    retention_time, spectra, wavelength = [], [], None
    for index, (start, stop) in enumerate(zip(starts, starts[1:])):
        retention_time.append(struct.unpack("<d", data[start + 19:start + 27])[0])
        start_wavelength = struct.unpack("<d", data[start + 54:start + 62])[0]
        values = _sfd_varints(data[start + SFD_HEADER_SIZE:stop])
        try:
            end = values.index(SFD_END_OF_SPECTRUM, 4)
        except ValueError:
            raise ValueError(
                f"Spectrum {index} in {name!r} has no end marker; the file may be "
                f"damaged or use an unknown format."
            ) from None

        step = _sfd_signed(values[0]) / 10
        scale = _sfd_signed(values[2]) / _sfd_signed(values[3])
        counts = np.cumsum([_sfd_signed(v) for v in values[4:end]])

        if wavelength is None:
            wavelength = start_wavelength + step * np.arange(len(counts))
        elif len(counts) != len(wavelength):
            raise ValueError(
                f"Spectrum {index} in {name!r} has {len(counts)} wavelengths, "
                f"expected {len(wavelength)}."
            )
        spectra.append(counts * scale)

    return np.array(retention_time), wavelength, np.array(spectra)


def _datetime_from_raw_data_id(raw_data_id):
    r"""Read the creation time from an id like '2026\10\21\121015935.sfd'.

    Chromeleon names the file after the moment it was created, which is
    0-1.1 s after the injection (checked on 11 injections). Returned to the
    second, as instrument local time.
    """
    match = re.search(r"(\d{4})\\(\d{1,2})\\(\d{1,2})\\(\d{2})(\d{2})(\d{2})\d*\.sfd$", raw_data_id or "")
    if not match:
        return None
    return datetime(*(int(part) for part in match.groups()))


def read_cmbx(path):
    """Read every injection with a 3D field from a .cmbx archive into RawRuns.

    Internal reader; users call DADSequence.from_cmbx, which documents it.
    Injections without a 3D field (e.g. a wash) are skipped.
    """
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise ValueError(f"{str(path)!r} is not a Chromeleon .cmbx archive (not a ZIP file).") from None

    runs = []
    with archive:
        try:
            root = ET.fromstring(archive.read("header.xml"))
        except KeyError:
            raise ValueError(f"No header.xml in {str(path)!r}; is it a Chromeleon .cmbx archive?") from None

        sequence = next(
            (el.get("Name") for el in root.iter("ChromeleonElement")
             if el.get("ItemType") == "Dionex.Chromeleon.Data.Sequence"),
            None,
        )
        injections = [el for el in root.iter("ChromeleonElement") if el.get("ItemType") == CMBX_INJECTION_TYPE]

        for number, injection in enumerate(injections, start=1):
            fields_3d = [s for s in injection if s.get("RawDataFilename", "").endswith(".sfd")]
            if not fields_3d:
                continue
            field = fields_3d[0]
            name = injection.get("Name")
            file_name = field.get("RawDataFilename")

            injection_datetime = _datetime_from_raw_data_id(field.get("RawDataFileId"))
            if injection_datetime is None:
                raise ValueError(
                    f"Could not read the injection time of {name!r} from "
                    f"{field.get('RawDataFileId')!r} in {str(path)!r}."
                )
            retention_time, wavelength, signal = _read_sfd(archive.read(file_name), file_name)

            meta = DADMeta(
                sample_name=name,
                sequence=sequence,
                injection_datetime=injection_datetime,
                injection_number=number,
                source_path=str(path),
                source_format="chromeleon-cmbx",
                injection_id=_injection_id_from_url(injection.get("Url")),
            )
            runs.append(
                RawRun(
                    rt=retention_time,
                    wl=wavelength,
                    signal=signal,
                    meta=meta,
                    time_unit="min",
                    signal_unit="mAU",
                    rt_as_rows=True,
                )
            )

    if not runs:
        raise ValueError(f"No injection with a 3D field found in {str(path)!r}.")
    return runs