"""Reader for Chromeleon 3D field text exports (tab-separated ASCII)."""
import io
import re
from datetime import datetime

import numpy as np
import pandas as pd

from ..meta import DADMeta
from ..raw import RawRun

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
        time = table.iloc[:, 0].to_numpy(dtype=np.float64)
        intensity = table.iloc[:, 2:].to_numpy(dtype=np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError("Raw data block contains non-numeric fields.") from e

    if np.isnan(time).any() or np.isnan(intensity).any():
        raise ValueError("Raw data block contains empty or non-numeric fields.")

    return time, intensity


def read_chromeleon(path, *, time_unit="min", intensity_unit="mAU", max_metadata_lines=100):
    """Parse a Chromeleon 3D field export into a RawRun, with no cleanup.

    Internal reader; users call DADRun.from_chromeleon, which documents the
    options. time_unit and intensity_unit label the units used in the file.

    Both the older export layout (comma decimals, 'Injection ...' metadata
    fields) and the newer one (point decimals, 'Name' / 'Inject Time' fields,
    'n.a.' integration times) are supported.
    """
    metadata = {}
    marker_found = False
    time, intensity_rows = [], []
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
                time, intensity_rows = _parse_data_block(
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
    if len(time) == 0:
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
        time=np.array(time),
        wl=np.array(wavelength),
        intensity=np.array(intensity_rows),
        meta=meta,
        time_unit=time_unit,
        intensity_unit=intensity_unit,
        time_as_rows=True,
    )


def _injection_id_from_url(url):
    """Return Chromeleon's injection id (e.g. '556.smp') from a chrom:// URL, or None."""
    if not url:
        return None
    match = re.search(r"/([^/]+\.smp)(?:/|$)", url)
    return match.group(1) if match else None
