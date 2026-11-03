from dataclasses import dataclass
import numpy as np
from .meta import DADMeta

TIME_UNITS_TO_MIN = {
    "min": 1.0,
    "s": 1 / 60,
    "h": 60.0,
}
SIGNAL_UNITS_TO_MAU = {
    "mAU": 1.0,
    "AU": 1e3,
}


@dataclass(frozen=True)
class RawRun:
    """Raw data prior to validation as returned by a suitable reader.

    signal_unit must be mAU or AU.
    time_unit must be min, s, or h. 
    """

    rt: np.ndarray
    wl: np.ndarray
    signal: np.ndarray
    meta: DADMeta
    time_unit: str
    signal_unit: str
    rt_as_rows: bool | None = None

    def __post_init__(self):
        if self.time_unit not in TIME_UNITS_TO_MIN:
            raise ValueError(
                f"Unknown time unit {self.time_unit!r}. "
                f"Known units: {list(TIME_UNITS_TO_MIN)}."
            )
        if self.signal_unit not in SIGNAL_UNITS_TO_MAU:
            raise ValueError(
                f"Unknown signal unit {self.signal_unit!r}. "
                f"Known units: {list(SIGNAL_UNITS_TO_MAU)}."
            )