# hplcdad

Load and work with HPLC-DAD (diode array detector) data in Python.

`hplcdad` reads vendor export files into a single, clean data structure: a signal matrix with a time axis, a wavelength axis, and run metadata. Whichever instrument a file came from, the rest of your analysis code sees the same layout and units.

## Supported formats

| Vendor | Format | Method |
|---|---|---|
| Thermo Chromeleon (7.2.7) | 3D field export, tab-separated ASCII (`3DFIELD`) | `DADRun.from_chromeleon(path)` |

More formats can be added; see [Adding a new format](#adding-a-new-format).

## Installation

Requires Python 3.10 or newer and NumPy.

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -e .
```

The `-e` flag installs the package in editable mode, so changes to the source take effect without reinstalling.

## Quick start

```python
import numpy as np
from hplcdad import DADRun

run = DADRun.from_chromeleon("path/to/export.txt")

print(run.meta.sample_name, run.meta.injection_datetime)
print(run.signal.shape)   # (number of time points, number of wavelengths)

# Chromatogram at 254 nm
col = np.argmin(np.abs(run.wl - 254))
chromatogram = run.signal[:, col]

# Spectrum at 2.0 min
row = np.argmin(np.abs(run.rt - 2.0))
spectrum = run.signal[row, :]
```

## The `DADRun` object

| Attribute | Description |
|---|---|
| `rt` | Time axis, 1D array, in minutes |
| `wl` | Wavelength axis, 1D array, in nm |
| `signal` | Intensity matrix, shape `(len(rt), len(wl))` (time × wavelength), in mAU |
| `meta` | A `DADMeta` object (see below) |

Every `DADRun` is guaranteed to have these units and this layout, however it was created. The constructor raises a `ValueError` if the signal shape does not match the axes.

### Metadata

`run.meta` is a frozen (read-only) dataclass:

| Field | Description |
|---|---|
| `sample_name` | Sample or injection name |
| `injection_datetime` | `datetime` of the injection (instrument local time) |
| `injection_number` | Injection number, if available |
| `injection_volume_ul` | Injection volume in µL, if available |
| `vial_position` | Vial or tray position, if available |
| `method` | Instrument method name, if available |
| `source_path` | Path of the file the run was loaded from |
| `source_format` | Name of the format (e.g. `"chromeleon"`) |

Fields the file does not provide are `None`.

## How it works

Loading happens in three stages, so that the vendor-specific code stays in one place:

```
file  ->  reader  ->  RawRun  ->  normalise  ->  DADRun
```

1. **Reader** (`readers.py`): parses one vendor format and returns a `RawRun`, the data exactly as stored, together with its unit labels and an orientation hint. It does no cleanup.
2. **Normalise** (`processing.py`): converts units to minutes and mAU, makes sure the matrix is time × wavelength, and validates the result (matching shapes, strictly increasing axes, no NaN or infinite values).
3. **`DADRun`** (`run.py`): holds the clean data.

Errors at any stage are raised as `ValueError` with a message that names the problem and, where useful, the actual and expected values.

### Recognized units

Readers must label their data with one of these units:

| Quantity | Units |
|---|---|
| Time | `min`, `s`, `h` |
| Signal | `mAU`, `AU` |

An unknown label raises a `ValueError` when the `RawRun` is created. To support a new unit, add it to the tables in `raw.py`.

## Adding a new format

1. In `readers.py`, write `read_<vendor>(path)` that parses the file and returns a `RawRun` (signal matrix, time axis, wavelength axis, a `DADMeta`, unit labels, and `rt_as_rows` if the orientation is known).
2. In `run.py`, add a class method `from_<vendor>(cls, path)` that calls your reader and then `cls.from_raw(...)`.

The cleanup is shared, so a new reader only has to parse its own format.