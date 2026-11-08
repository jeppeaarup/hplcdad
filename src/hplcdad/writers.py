"""Write DADRuns as Chromeleon 3D field text exports.

Uses the layout Chromeleon 7.3.2 writes: metadata blocks, then "Raw Data:"
with one row per spectrum; '.' decimals and ',' thousands separators,
6 decimals, CRLF line endings, UTF-8 with a byte order mark. Other software
that reads Chromeleon exports can read these files, and
DADRun.from_chromeleon reads them back.
"""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DECIMALS = 6
DEFAULT_TIMEZONE = "Europe/Copenhagen"


def _package_version():
    try:
        return version("hplcdad")
    except PackageNotFoundError:
        return "unknown"


def _number(value, decimals=DECIMALS):
    """Format like Chromeleon: '.' decimals, ',' thousands, fixed decimals."""
    return f"{value:,.{decimals}f}"


def _wavelength(value):
    """190 -> '190.0'; keeps extra decimals only when needed (e.g. 190.25)."""
    text = f"{value:.6f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def _inject_time(moment, timezone):
    """Format like Chromeleon's 'Inject Time', e.g. '10/21/2026 1:25:22 PM +02:00'.

    DADMeta holds instrument local time without a time zone, so the UTC
    offset is worked out from ``timezone`` for that date (summer time
    included).
    """
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        raise ValueError(
            f"Unknown time zone {timezone!r}; use an IANA name such as 'Europe/Copenhagen'."
        ) from None
    offset_minutes = int(moment.replace(tzinfo=zone).utcoffset().total_seconds() // 60)
    sign = "+" if offset_minutes >= 0 else "-"
    hours, minutes = divmod(abs(offset_minutes), 60)

    hour = moment.hour % 12 or 12
    half = "AM" if moment.hour < 12 else "PM"
    return (
        f"{moment.month}/{moment.day}/{moment.year} "
        f"{hour}:{moment.minute:02d}:{moment.second:02d} {half} "
        f"{sign}{hours:02d}:{minutes:02d}"
    )


def _text(value):
    return "" if value is None else str(value)


def write_chromeleon(run, path, *, overwrite=False, timezone=DEFAULT_TIMEZONE):
    """Write a DADRun as a Chromeleon 3D field text export. Internal; users call DADRun.to_chromeleon."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{str(path)!r} already exists; pass overwrite=True to replace it.")

    meta = run.meta
    url = (
        f"chrom://hplcdad/{meta.sequence}.seq/{meta.injection_id}/DAD_Spectrum.field"
        if meta.injection_id
        else ""
    )
    volume = "" if meta.injection_volume_ul is None else _number(meta.injection_volume_ul, 4)
    generated_by = f"hplcdad, from {meta.source_format}" if meta.source_format else "hplcdad"

    lines = [
        f"URL\t{url}",
        "Spectral Field\tDAD_Spectrum",
        "",
        "Injection Information:",
        "Data Vault\t",
        f"Sequence\t{_text(meta.sequence)}",
        f"Name\t{meta.sample_name}",
        f"Number\t{_text(meta.injection_number)}",
        f"Position\t{_text(meta.vial_position)}",
        "Comment\t",
        "Processing Method\t",
        f"Instrument Method\t{_text(meta.method)}",
        "Type\t",
        "Status\t",
        f"Inject Time\t{_inject_time(meta.injection_datetime, timezone)}",
        f"Volume (µL)\t{volume}",
        "Dilution Factor\t",
        "Weight\t",
        "",
        "Raw Data Information:",
        f"Time Min. (min)\t{_number(run.time[0])}",
        f"Time Max. (min)\t{_number(run.time[-1])}",
        f"Scan Min. (nm)\t{_number(run.wavelength[0])}",
        f"Scan Max. (nm)\t{_number(run.wavelength[-1])}",
        f"Intensity Min. (mAU)\t{_number(run.intensity.min())}",
        f"Intensity Max. (mAU)\t{_number(run.intensity.max())}",
        f"Spectra\t{len(run.time)}",
        "Detector\t",
        "Detector Type\tUV",
        f"Generating Data System\t{generated_by}",
        f"Exporting Data System\thplcdad {_package_version()}",
        "Spectral Field\tDAD_Spectrum",
        "Driver Name\t",
        "",
        "Raw Data:",
        "Time (min)\tIntegr.Time (s)\t" + "\t".join(_wavelength(w) for w in run.wavelength),
    ]

    number = f"{{:,.{DECIMALS}f}}".format
    for time, spectrum in zip(run.time, run.intensity):
        lines.append(f"{time:.{DECIMALS}f}\tn.a.\t" + "\t".join(map(number, spectrum)))

    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8-sig"))
    return path
