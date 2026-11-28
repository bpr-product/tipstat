"""Date primitives."""
from datetime import date, datetime


def valid_datetime(s) -> str | None:
    """YYYY-MM-DDTHH:MM (local wall time) when the text carries a time of day, else None."""
    try:
        d = datetime.fromisoformat(str(s).strip().replace("Z", "")[:16])
        return d.strftime("%Y-%m-%dT%H:%M") if "T" in str(s) or " " in str(s).strip() else None
    except ValueError:
        return None


def valid_date(s) -> str | None:
    """ISO date (YYYY-MM-DD) from the start of s, or None. Unknown dates stay unknown."""
    try:
        return date.fromisoformat(str(s)[:10]).isoformat()
    except ValueError:
        return None
