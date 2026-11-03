# hplcdad

Load and work with HPLC-DAD (diode array detector) data in Python.

`hplcdad` reads vendor files into a single, clean data structure: a signal matrix with a time axis, a wavelength axis, and run metadata. Whichever instrument or format a file came from, the rest of your analysis code sees the same layout and units. Runs can be saved as HDF5 files, a common open format that keeps data and metadata together.

## Supported formats

| Vendor | Format | Method |
|---|---|---|
| Thermo Chromeleon (7.2.7, 7.3.2) | 3D field export, tab-separated ASCII (`3DFIELD`) | `DADRun.from_chromeleon(path)`, `DADSequence.from_chromeleon(folder)` |
| Thermo Chromeleon (7.3.2) | Backup archive (`.cmbx`), read directly with no export needed | `DADSequence.from_cmbx(path)` |
| hplcdad | HDF5 file written by `to_hdf5` | `DADRun.from_hdf5(path)`, `DADSequence.from_hdf5(folder)` |

More formats can be added; see [Adding a new format](#adding-a-new-format).

## Installation

Requires Python 3.10 or newer, NumPy and pandas. Reading and writing HDF5 files also needs h5py.

```bash
git clone https://github.com/<your-username>/hplcdad.git
cd hplcdad
pip install -e ".[hdf5]"
```

The `-e` flag installs the package in editable mode, so changes to the source take effect without reinstalling. Leave out `[hdf5]` if you don't need HDF5 files; use `[hdf5,dev]` to also install the testing tools.

## Quick start

```python
from hplcdad import DADRun, DADSequence

run = DADRun.from_chromeleon("path/to/export.txt")

print(run.meta.sample_name, run.meta.injection_datetime)
print(run.signal.shape)   # (number of time points, number of wavelengths)

chromatogram = run.get_chromatogram(254)   # nearest wavelength to 254 nm
spectrum = run.get_spectrum(2.0)           # nearest time point to 2.0 min
```

Read a whole sequence straight from a Chromeleon backup, and save it as HDF5:

```python
seq = DADSequence.from_cmbx("path/to/sequence.cmbx")
print(seq)                  # one line per run
seq.to_hdf5("converted/")   # one .h5 file per run

seq = DADSequence.from_hdf5("converted/")
```

## The `DADRun` object

| Attribute | Description |
|---|---|
| `rt` | Time axis, 1D array, in minutes |
| `wl` | Wavelength axis, 1D array, in nm |
| `signal` | Intensity matrix, shape `(len(rt), len(wl))` (time × wavelength), in mAU |
| `meta` | A `DADMeta` object (see below) |

Every `DADRun` is guaranteed to have these units and this layout, however it was created. The constructor raises a `ValueError` if the signal shape does not match the axes.

`DADSequence` is an ordered, read-only collection of runs. It can be indexed by position or sample name, filtered and sorted on metadata, and stacked into one array for multivariate analysis.

### Metadata

`run.meta` is a frozen (read-only) dataclass:

| Field | Description |
|---|---|
| `sample_name` | Sample or injection name |
| `sequence` | Name of the sequence the injection belongs to |
| `injection_datetime` | `datetime` of the injection (instrument local time) |
| `injection_number` | Injection number, if available |
| `injection_volume_ul` | Injection volume in µL, if available |
| `vial_position` | Vial or tray position, if available |
| `method` | Instrument method name, if available |
| `source_path` | Path of the file the run was loaded from |
| `source_format` | Name of the format (e.g. `"chromeleon"`, `"chromeleon-cmbx"`) |
| `injection_id` | Chromeleon's own injection id (e.g. `"556.smp"`), unique even when sample names repeat |

Fields the file does not provide are `None`.

## Chromeleon backup archives (`.cmbx`)

`DADSequence.from_cmbx` reads the 3D field of every injection in a `.cmbx` archive, so no manual export from Chromeleon is needed. Injections without a 3D field (e.g. a wash) are skipped.

- **Data:** identical to Chromeleon's own 3D field text export, to within the export's 6-decimal rounding.
- **Metadata:** more limited than in a text export.
  - `injection_datetime` comes from the name Chromeleon gives the 3D field file, which is 0–1.1 s after the actual injection.
  - `injection_number` is the injection's position in the sequence.
  - `injection_volume_ul`, `vial_position` and `method` are `None`.
- **Validated on:** Chromeleon 7.3.2 data recorded at 190–640 nm, 2 nm steps and 2.5 Hz, compared against text exports of the same injections. The `.cmbx` format is undocumented and was worked out from the files themselves, so check one text export against the reader after changing the DAD settings or updating Chromeleon.

## HDF5 files

`to_hdf5` saves each run as one HDF5 file:

| Path | Content |
|---|---|
| `/rt` | Time axis, float64, `units="min"` |
| `/wl` | Wavelength axis, float64, `units="nm"` |
| `/signal` | Signal, `(rt, wl)`, float64, `units="mAU"`, gzip-compressed |
| root attributes | All `DADMeta` fields that are not `None`, plus `hplcdad_schema_version`, `created` (UTC) and `source_sha256` (checksum of the source file, when it still exists) |

A run read back with `from_hdf5` is identical to the one written. The checksum ties each file to the raw data it came from. The files use HDF5 dimension scales, so they also open in xarray with labelled axes:

```python
import xarray as xr
ds = xr.open_dataset("converted/BLANK-01.h5", engine="h5netcdf")
```

`DADSequence.to_hdf5` names files after the sample; when names repeat, the injection id is added (`OPTI_01 (836.smp).h5`). Existing files are never overwritten unless you pass `overwrite=True`.

## Writing Chromeleon text exports

`to_chromeleon` writes a run in the layout of Chromeleon's 3D field text export, for software that reads those files:

```python
run.to_chromeleon("exports/BLANK-01.txt")
seq.to_chromeleon("exports/")      # one .txt file per run
```

Values are written with 6 decimals, as Chromeleon does. Metadata the run does not have (e.g. volume and vial position for runs read from a `.cmbx`) are left empty, and the injection time is written without a UTC offset. `DADRun.from_chromeleon` reads the files back.

## How it works

Loading happens in three stages, so that the vendor-specific code stays in one place:

```
file  ->  reader  ->  RawRun  ->  normalise  ->  DADRun
```

1. **Reader** (a module in `readers/`, or `hdf5.py`): parses one format and returns a `RawRun` (or a list of them, for files that hold a whole sequence, such as `.cmbx`): the data exactly as stored, together with its unit labels and an orientation hint. It does no cleanup.
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

1. Add a module `readers/<format>.py` with a function `read_<format>(path)` that parses the file and returns a `RawRun` (signal matrix, time axis, wavelength axis, a `DADMeta`, unit labels, and `rt_as_rows` if the orientation is known). If one file holds several runs, return a list of `RawRun`. Then import it in `readers/__init__.py`.
2. Add a class method that calls your reader and then `DADRun.from_raw(...)`: `DADRun.from_<vendor>(path)` for one run per file, or `DADSequence.from_<vendor>(path)` for several.

The cleanup is shared, so a new reader only has to parse its own format. Once loaded, any run can be saved with `to_hdf5`.

## Project layout

```
pyproject.toml
src/hplcdad/
  __init__.py     public classes
  readers/        one module per file format (chromeleon_txt.py, chromeleon_cmbx.py)
  writers.py      Chromeleon text export writer
  hdf5.py         HDF5 reading and writing
  raw.py          RawRun and unit tables
  processing.py   unit conversion, orientation, validation
  run.py          DADRun
  sequence.py     DADSequence
  traces.py       Chromatogram and Spectrum
  preprocessing.py  AsLS baseline and Savitzky-Golay filter
  meta.py         DADMeta
tests/
```