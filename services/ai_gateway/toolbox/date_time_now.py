from datetime import datetime


def get_current_datetime() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()
