"""A small window for converting HPLC-DAD files between formats.

Start it with ``hplcdad-gui`` (installed with the package) or
``python -m hplcdad.gui``. It wraps the same readers and writers as the
Python API: choose a file or folder, an output format and a folder, and
press Convert.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import traceback
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .sequence import DADSequence

# key: (label, file suffix)
INPUT_FORMATS = {
    "chromeleon": ("Chromeleon text export (.txt)", ".txt"),
    "cmbx": ("Chromeleon backup (.cmbx)", ".cmbx"),
    "hdf5": ("hplcdad HDF5 (.h5)", ".h5"),
}
OUTPUT_FORMATS = {
    "hdf5": ("HDF5 (.h5)", ".h5"),
    "chromeleon": ("Chromeleon text export (.txt)", ".txt"),
}
AUTO = "Auto-detect"
DEFAULT_TIMEZONE = "Europe/Copenhagen"
COMMON_TIMEZONES = (
    "Europe/Copenhagen", "Europe/London", "Europe/Berlin", "UTC",
    "America/New_York", "America/Chicago", "America/Los_Angeles", "Asia/Tokyo",
)


class Cancelled(Exception):
    """Raised from a progress callback when the user presses Cancel."""


# --------------------------------------------------------------- conversion
# Plain functions, independent of the window, so they can be tested.

def detect_format(source) -> str | None:
    """Guess the input format of a file or folder from file extensions.

    For a folder, the most common supported extension among its files wins.
    Returns a key of INPUT_FORMATS, or None if nothing matches.
    """
    path = Path(source)
    by_suffix = {suffix: key for key, (_, suffix) in INPUT_FORMATS.items()}
    if path.is_file():
        return by_suffix.get(path.suffix.lower())
    if path.is_dir():
        counts = Counter(
            p.suffix.lower() for p in path.iterdir()
            if p.is_file() and p.suffix.lower() in by_suffix
        )
        if counts:
            return by_suffix[counts.most_common(1)[0][0]]
    return None


def default_output_folder(source, output_format: str) -> Path:
    """Where to write by default: next to the input, named after it.

    A single text or HDF5 file is converted into its own folder; a .cmbx
    archive or a folder gets a sibling folder like "<name> (h5)".
    """
    path = Path(source)
    suffix = OUTPUT_FORMATS[output_format][1].lstrip(".")
    if path.is_file() and path.suffix.lower() != ".cmbx":
        return path.parent
    name = path.stem if path.is_file() else path.name
    return path.parent / f"{name} ({suffix})"


def load(source, input_format: str, progress=None) -> DADSequence:
    """Read a file or folder in ``input_format`` into a DADSequence."""
    if input_format == "chromeleon":
        return DADSequence.from_chromeleon(source, progress=progress)
    if input_format == "cmbx":
        return DADSequence.from_cmbx(source, progress=progress)
    if input_format == "hdf5":
        return DADSequence.from_hdf5(source, progress=progress)
    raise ValueError(f"Unknown input format {input_format!r}.")


def save(sequence: DADSequence, folder, output_format: str, *, overwrite=False,
         timezone=DEFAULT_TIMEZONE, progress=None) -> list[Path]:
    """Write every run of ``sequence`` to ``folder`` in ``output_format``."""
    if output_format == "hdf5":
        return sequence.to_hdf5(folder, overwrite=overwrite, progress=progress)
    if output_format == "chromeleon":
        return sequence.to_chromeleon(
            folder, overwrite=overwrite, timezone=timezone, progress=progress
        )
    raise ValueError(f"Unknown output format {output_format!r}.")


# ------------------------------------------------------------------- window

def main():
    """Open the converter window."""
    import tkinter as tk

    try:   # sharp text on high-DPI Windows screens
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass

    root = tk.Tk()
    ConverterApp(root)
    root.mainloop()


class ConverterApp:
    POLL_MS = 50

    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText

        self.tk, self.ttk = tk, ttk
        self.root = root
        root.title("hplcdad converter")
        root.minsize(620, 460)

        self.messages = queue.Queue()
        self.cancel_event = threading.Event()
        self.worker = None
        self.suggested_output = ""
        self.last_output = None

        self.source_var = tk.StringVar()
        self.input_format_var = tk.StringVar(value=AUTO)
        self.detected_var = tk.StringVar()
        self.output_format_var = tk.StringVar(value="hdf5")
        self.timezone_var = tk.StringVar(value=DEFAULT_TIMEZONE)
        self.output_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Choose a file or folder to convert.")

        frame = ttk.Frame(root, padding=12)
        frame.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(8, weight=1)
        pad = {"padx": 4, "pady": 4}

        # Input
        ttk.Label(frame, text="Input:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.source_var).grid(row=0, column=1, sticky="ew", **pad)
        buttons = ttk.Frame(frame)
        buttons.grid(row=0, column=2, sticky="e")
        ttk.Button(buttons, text="File…", command=self.choose_file).pack(side="left", padx=2)
        ttk.Button(buttons, text="Folder…", command=self.choose_folder).pack(side="left", padx=2)
        self.source_var.trace_add("write", lambda *_: self.on_source_changed())

        ttk.Label(frame, text="Input format:").grid(row=1, column=0, sticky="w", **pad)
        format_box = ttk.Combobox(
            frame, textvariable=self.input_format_var, state="readonly",
            values=[AUTO] + [label for label, _ in INPUT_FORMATS.values()],
        )
        format_box.grid(row=1, column=1, sticky="w", **pad)
        format_box.bind("<<ComboboxSelected>>", lambda _: self.on_source_changed())
        ttk.Label(frame, textvariable=self.detected_var, foreground="gray").grid(
            row=1, column=2, sticky="w", **pad
        )

        # Output
        ttk.Label(frame, text="Convert to:").grid(row=2, column=0, sticky="w", **pad)
        radios = ttk.Frame(frame)
        radios.grid(row=2, column=1, columnspan=2, sticky="w", **pad)
        for key, (label, _) in OUTPUT_FORMATS.items():
            ttk.Radiobutton(
                radios, text=label, value=key, variable=self.output_format_var,
                command=self.on_output_format_changed,
            ).pack(side="left", padx=(0, 12))

        ttk.Label(frame, text="Time zone:").grid(row=3, column=0, sticky="w", **pad)
        self.timezone_box = ttk.Combobox(
            frame, textvariable=self.timezone_var, values=COMMON_TIMEZONES
        )
        self.timezone_box.grid(row=3, column=1, sticky="w", **pad)
        ttk.Label(
            frame, text="Instrument time zone, for Chromeleon output", foreground="gray"
        ).grid(row=3, column=2, sticky="w", **pad)

        ttk.Label(frame, text="Output folder:").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.output_var).grid(row=4, column=1, sticky="ew", **pad)
        ttk.Button(frame, text="Browse…", command=self.choose_output).grid(
            row=4, column=2, sticky="w", **pad
        )

        # Actions
        actions = ttk.Frame(frame)
        actions.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 4))
        self.convert_button = ttk.Button(actions, text="Convert", command=self.start)
        self.convert_button.pack(side="left", padx=4)
        self.cancel_button = ttk.Button(
            actions, text="Cancel", command=self.cancel, state="disabled"
        )
        self.cancel_button.pack(side="left", padx=4)
        self.open_button = ttk.Button(
            actions, text="Open output folder", command=self.open_output, state="disabled"
        )
        self.open_button.pack(side="right", padx=4)

        # Progress
        self.progress = ttk.Progressbar(frame, mode="determinate", maximum=1)
        self.progress.grid(row=6, column=0, columnspan=3, sticky="ew", **pad)
        ttk.Label(frame, textvariable=self.status_var).grid(
            row=7, column=0, columnspan=3, sticky="w", **pad
        )

        # Log
        self.log_box = ScrolledText(frame, height=10, state="disabled", wrap="word")
        self.log_box.grid(row=8, column=0, columnspan=3, sticky="nsew", **pad)

        self.on_output_format_changed()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------ choosing

    def choose_file(self):
        from tkinter import filedialog
        suffixes = " ".join(f"*{s}" for _, s in INPUT_FORMATS.values())
        path = filedialog.askopenfilename(
            title="Choose a file to convert",
            initialdir=self._initial_dir(),
            filetypes=[("Supported files", suffixes)]
            + [(label, f"*{s}") for label, s in INPUT_FORMATS.values()]
            + [("All files", "*.*")],
        )
        if path:
            self.source_var.set(str(Path(path)))

    def choose_folder(self):
        from tkinter import filedialog
        path = filedialog.askdirectory(
            title="Choose a folder of files to convert", initialdir=self._initial_dir()
        )
        if path:
            self.source_var.set(str(Path(path)))

    def choose_output(self):
        from tkinter import filedialog
        path = filedialog.askdirectory(
            title="Choose the output folder", initialdir=self._initial_dir()
        )
        if path:
            self.output_var.set(str(Path(path)))

    def _initial_dir(self):
        source = Path(self.source_var.get().strip())
        if source.is_file():
            return str(source.parent)
        if source.is_dir():
            return str(source)
        return None

    def input_format(self) -> str | None:
        chosen = self.input_format_var.get()
        if chosen == AUTO:
            return detect_format(self.source_var.get().strip())
        return next(key for key, (label, _) in INPUT_FORMATS.items() if label == chosen)

    def on_source_changed(self):
        source = self.source_var.get().strip()
        if not source or not Path(source).exists():
            self.detected_var.set("")
            return
        if self.input_format_var.get() == AUTO:
            key = detect_format(source)
            self.detected_var.set(
                f"Detected: {INPUT_FORMATS[key][0]}" if key else "No supported files found"
            )
        else:
            self.detected_var.set("")
        self._suggest_output()

    def on_output_format_changed(self):
        chromeleon = self.output_format_var.get() == "chromeleon"
        self.timezone_box.configure(state="normal" if chromeleon else "disabled")
        self._suggest_output()

    def _suggest_output(self):
        """Fill in a default output folder, unless the user typed their own."""
        source = self.source_var.get().strip()
        current = self.output_var.get().strip()
        if not source or not Path(source).exists():
            return
        if current and current != self.suggested_output:
            return
        self.suggested_output = str(
            default_output_folder(source, self.output_format_var.get())
        )
        self.output_var.set(self.suggested_output)

    # ------------------------------------------------------------- running

    def start(self):
        from tkinter import messagebox

        source = self.source_var.get().strip()
        output = self.output_var.get().strip()
        if not source or not Path(source).exists():
            messagebox.showerror("hplcdad", "Choose an input file or folder that exists.")
            return
        input_format = self.input_format()
        if input_format is None:
            messagebox.showerror(
                "hplcdad",
                "Could not tell the input format from the file extensions.\n"
                "Choose the input format from the list.",
            )
            return
        if not output:
            messagebox.showerror("hplcdad", "Choose an output folder.")
            return
        output_format = self.output_format_var.get()
        timezone = self.timezone_var.get().strip()
        if output_format == "chromeleon":
            try:
                ZoneInfo(timezone)
            except (ZoneInfoNotFoundError, ValueError):
                messagebox.showerror(
                    "hplcdad",
                    f"Unknown time zone {timezone!r}.\nUse an IANA name such as "
                    f"'Europe/Copenhagen' or 'UTC'.",
                )
                return

        self.log(
            f"Converting {source}\n  {INPUT_FORMATS[input_format][0]} → "
            f"{OUTPUT_FORMATS[output_format][0]}\n  into {output}"
        )
        job = dict(output=output, output_format=output_format, timezone=timezone)
        self._run_worker(self._convert_job, source, input_format, job)

    def _run_worker(self, target, *args):
        self.cancel_event.clear()
        self.convert_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.open_button.configure(state="disabled")
        self.worker = threading.Thread(target=target, args=args, daemon=True)
        self.worker.start()
        self.root.after(self.POLL_MS, self.poll)

    def _progress_callback(self, step):
        def callback(done, total):
            if self.cancel_event.is_set():
                raise Cancelled
            self.messages.put(("progress", step, done, total))
        return callback

    def _convert_job(self, source, input_format, job):
        """Runs in the worker thread; talks to the window only via the queue."""
        try:
            self.messages.put(("progress", "Reading", 0, 0))
            sequence = load(source, input_format, progress=self._progress_callback("Reading"))
            self.messages.put(("log", f"Read {len(sequence)} run(s)."))
        except Cancelled:
            self.messages.put(("cancelled",))
            return
        except Exception as e:
            self.messages.put(("error", _describe(e)))
            return
        self._save_job(sequence, job, overwrite=False)

    def _save_job(self, sequence, job, overwrite):
        try:
            paths = save(
                sequence, job["output"], job["output_format"], overwrite=overwrite,
                timezone=job["timezone"], progress=self._progress_callback("Writing"),
            )
        except Cancelled:
            self.messages.put(("cancelled",))
        except FileExistsError as e:
            self.messages.put(("exists", str(e), sequence, job))
        except Exception as e:
            self.messages.put(("error", _describe(e)))
        else:
            self.messages.put(("done", paths, job["output"]))

    def poll(self):
        """Handle messages from the worker; runs in the window's thread."""
        from tkinter import messagebox

        finished = False
        try:
            while True:
                message = self.messages.get_nowait()
                kind = message[0]
                if kind == "progress":
                    self._show_progress(*message[1:])
                elif kind == "log":
                    self.log(message[1])
                elif kind == "done":
                    paths, folder = message[1], message[2]
                    self.last_output = folder
                    self.log(f"Wrote {len(paths)} file(s) to {folder}\n")
                    self._finish(f"Done: {len(paths)} file(s) written.", complete=True)
                    finished = True
                elif kind == "cancelled":
                    self.log("Cancelled. Files written so far are kept.\n")
                    self._finish("Cancelled.")
                    finished = True
                elif kind == "error":
                    self.log(f"Error: {message[1]}\n")
                    self._finish("Failed; see the log below.")
                    messagebox.showerror("hplcdad", message[1])
                    finished = True
                elif kind == "exists":
                    text, sequence, job = message[1:]
                    if messagebox.askyesno(
                        "hplcdad",
                        f"{text.split(',')[0]} in\n{job['output']}\n\nReplace them?",
                    ):
                        self.log("Replacing existing files.")
                        self._run_worker(self._save_job, sequence, job, True)
                        return   # the new worker restarts polling
                    self.log("Stopped: output files already exist.\n")
                    self._finish("Nothing written; files already exist.")
                    finished = True
        except queue.Empty:
            pass
        if not finished:
            self.root.after(self.POLL_MS, self.poll)

    def _show_progress(self, step, done, total):
        if total == 0:   # total not known yet (e.g. counting injections)
            if str(self.progress.cget("mode")) != "indeterminate":
                self.progress.configure(mode="indeterminate")
                self.progress.start(15)
            self.status_var.set(f"{step}…")
            return
        if str(self.progress.cget("mode")) != "determinate":
            self.progress.stop()
            self.progress.configure(mode="determinate")
        self.progress.configure(maximum=total, value=done)
        self.status_var.set(f"{step}: {done} of {total}")

    def _finish(self, status, complete=False):
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=1, value=1 if complete else 0)
        self.status_var.set(status)
        self.convert_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")
        if self.last_output and Path(self.last_output).is_dir():
            self.open_button.configure(state="normal")

    def cancel(self):
        self.cancel_event.set()
        self.cancel_button.configure(state="disabled")
        self.status_var.set("Cancelling after the current file…")

    def open_output(self):
        if not self.last_output:
            return
        if sys.platform == "win32":
            os.startfile(self.last_output)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", self.last_output])

    def on_close(self):
        self.cancel_event.set()
        self.root.destroy()

    def log(self, text):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")


def _describe(error: Exception) -> str:
    """A readable one-paragraph message for an error shown to the user."""
    if isinstance(error, ImportError) and "h5py" in str(error):
        return f"{error}\n\nInstall it with:  pip install h5py"
    if isinstance(error, (ValueError, FileNotFoundError, ImportError, OSError)):
        return str(error)
    return "".join(traceback.format_exception_only(error)).strip()


if __name__ == "__main__":
    main()
