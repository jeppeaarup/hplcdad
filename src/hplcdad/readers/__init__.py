"""One module per file format; each turns a file into RawRun objects.

To add a format, add a module here and import its reader below.
"""
from .chromeleon_txt import read_chromeleon
from .chromeleon_cmbx import read_cmbx
