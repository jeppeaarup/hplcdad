import io
import re
import struct
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd

from .meta import DADMeta
from .raw import RawRun
RAW_DATA_MARKER = "Raw Data:"

# Metadata keys differ between Chromeleon export layouts; the first match wins.
NAME_KEYS = ("Injection", "Name")
SEQUENCE_KEYS = ("Sequence",)
NUMBER_KEYS = ("Injection Number", "Number")
VOLUME_KEYS = ("Injection Volume (µl)", "Volume (µL)")


def _to_float(value_string, decimal_comma=True):
    """Convert a Chromeleon-formatted number.

    decimal_comma=True: '.' thousands, ',' decimal (older exports).
    decimal_comma=False: plain '.' decimal, "," thousands (values >= 1000 are written like 1,043.1).
    """
    value_string = value_string.strip()
    if decimal_comma:
        return float(value_string.replace(".", "").replace(",", "."))
    return float(value_string.replace(",", ""))


def _uses_decimal_comma(header_cells):
    """Detect the number format from the wavelength header cells."""
    return any("," in cell for cell in header_cells)


def _parse_metadata_line(line, metadata):
    parts = line.rstrip("\r\n").split("\t")
    if len(parts) == 2:
        key, value = parts
        metadata[key.strip()] = value.strip()


def _lookup(metadata, keys):
    for key in keys:
        if metadata.get(key):
            return metadata[key]
    return None


def _require(metadata, key, path):
    try:
        return metadata[key]
    except KeyError:
        raise ValueError(
            f"Expected metadata field {key!r} not found in {path!r}. "
            f"Available fields: {sorted(metadata)}"
        ) from None


def _require_any(metadata, keys, path):
    value = _lookup(metadata, keys)
    if value is None:
        raise ValueError(
            f"None of the expected metadata fields {list(keys)} found in "
            f"{path!r}. Available fields: {sorted(metadata)}"
        )
    return value


def _parse_datetime(metadata, path):
    # Older layout: separate date and time fields.
    if "Injection Date" in metadata and "Injection Time" in metadata:
        date_string = metadata["Injection Date"]
        time_string = metadata["Injection Time"]
        try:
            return datetime.strptime(
                f"{date_string} {time_string}", "%d-%m-%Y %H:%M:%S"
            )
        except ValueError as e:
            raise ValueError(
                f"Could not parse injection date/time "
                f"'{date_string} {time_string}' in {path!r}"
            ) from e

    # Newer layout: one field, e.g. '10/19/2026 3:21:47 PM +02:00'.
    # The UTC offset is dropped: injection_datetime is instrument local time.
    if "Inject Time" in metadata:
        text = metadata["Inject Time"]
        local_text = re.sub(r"\s*[+-]\d{2}:?\d{2}$", "", text)
        try:
            return datetime.strptime(local_text, "%m/%d/%Y %I:%M:%S %p")
        except ValueError as e:
            raise ValueError(
                f"Could not parse 'Inject Time' value {text!r} in {path!r}"
            ) from e

    raise ValueError(
        f"Expected either 'Injection Date' and 'Injection Time' or "
        f"'Inject Time' in {path!r}. Available fields: {sorted(metadata)}"
    )


def _parse_data_block(file, n_wavelengths, decimal_comma=True):
    expected_columns = n_wavelengths + 2  # time, integration time, then wavelengths

    text = file.read()  # the rest of the file, after the header row
    if not text.strip():
        return np.empty(0), np.empty((0, n_wavelengths))

    table = pd.read_csv(
        io.StringIO(text),
        sep="\t",
        header=None,
        decimal="," if decimal_comma else ".",
        thousands="." if decimal_comma else ",",
        na_values=["n.a."],  # the integration time column can be 'n.a.'
    )

    if table.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns in raw data rows, "
            f"got {table.shape[1]}."
        )

    # Column 1 (integration time) is never used, so it may be empty or 'n.a.'.
    try:
        retention_time = table.iloc[:, 0].to_numpy(dtype=np.float64)
        signal = table.iloc[:, 2:].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError("Raw data block contains non-numeric fields.") from e

    if np.isnan(retention_time).any() or np.isnan(signal).any():
        raise ValueError("Raw data block contains empty or non-numeric fields.")

    return retention_time, signal


def read_chromeleon(path, *, time_unit="min", signal_unit="mAU", max_metadata_lines=100):
    """Parse a Chromeleon 3D field export into a RawRun, with no cleanup.

    Internal reader; users call DADRun.from_chromeleon, which documents the
    options. time_unit and signal_unit label the units used in the file.

    Both the older export layout (comma decimals, 'Injection ...' metadata
    fields) and the newer one (point decimals, 'Name' / 'Inject Time' fields,
    'n.a.' integration times) are supported.
    """
    metadata = {}
    marker_found = False
    retention_time, signal_rows = [], []
    wavelength = []
    decimal_comma = True

    with open(path, encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if line.startswith(RAW_DATA_MARKER):
                marker_found = True
                header_line = next(file, None)
                if header_line is None:
                    raise ValueError(f"No header row after {RAW_DATA_MARKER!r} in {path!r}.")
                header_cells = header_line.strip().split("\t")[2:]
                decimal_comma = _uses_decimal_comma(header_cells)
                wavelength = [_to_float(h, decimal_comma) for h in header_cells]
                retention_time, signal_rows = _parse_data_block(
                    file, len(wavelength), decimal_comma
                )
                break

            _parse_metadata_line(line, metadata)

            if line_number >= max_metadata_lines:
                raise ValueError(
                    f"{RAW_DATA_MARKER!r} marker not found in the first "
                    f"{max_metadata_lines} lines of {path!r}. Is this a "
                    f"Chromeleon 3D field export?"
                )

    if not marker_found:
        raise ValueError(
            f"{RAW_DATA_MARKER!r} marker not found in {path!r} "
            f"(file ended before it appeared)."
        )
    if len(retention_time) == 0:
        raise ValueError(f"No data rows found after {RAW_DATA_MARKER!r} in {path!r}.")

    injection_number = _lookup(metadata, NUMBER_KEYS)
    injection_volume = _lookup(metadata, VOLUME_KEYS)

    meta = DADMeta(
        sample_name=_require_any(metadata, NAME_KEYS, path),
        sequence=_require_any(metadata, SEQUENCE_KEYS, path),
        injection_datetime=_parse_datetime(metadata, path),
        injection_number=int(injection_number) if injection_number else None,
        injection_volume_ul=(
            _to_float(injection_volume, decimal_comma) if injection_volume else None
        ),
        vial_position=metadata.get("Position") or None,
        method=metadata.get("Instrument Method") or None,
        source_path=str(path),
        source_format="chromeleon",
        injection_id=_injection_id_from_url(metadata.get("URL")),
    )

    return RawRun(
        rt=np.array(retention_time),
        wl=np.array(wavelength),
        signal=np.array(signal_rows),
        meta=meta,
        time_unit=time_unit,
        signal_unit=signal_unit,
        rt_as_rows=True,
    )


def _injection_id_from_url(url):
    """Return Chromeleon's injection id (e.g. '556.smp') from a chrom:// URL, or None."""
    if not url:
        return None
    match = re.search(r"/([^/]+\.smp)(?:/|$)", url)
    return match.group(1) if match else None


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