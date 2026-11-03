from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class DADMeta:
    """Metadata for one DAD run. injection_datetime is instrument local time."""

    sample_name: str
    sequence : str
    injection_datetime: datetime
    injection_number: int | None = None
    injection_volume_ul: float | None = None
    vial_position: str | None = None
    method: str | None = None
    source_path: str | None = None
    source_format: str | None = None
    injection_id: str | None = None  # Chromeleon's own id, e.g. "556.smp"; unique even when names repeat