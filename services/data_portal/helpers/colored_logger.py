# colored_logger.py
from __future__ import annotations

import datetime
import sys
from enum import Enum, auto
from typing import Any, Callable, Optional, Protocol, cast

from colorama import Fore, Style, init

# ----------------------------------------------------------------------
# Initialise colour handling (once per interpreter)
# ----------------------------------------------------------------------
init(autoreset=True)


# ----------------------------------------------------------------------
# Protocol for a formatter callable
# ----------------------------------------------------------------------
class _Formatter(Protocol):
    def __call__(self, fmt: str, /, *args: Any, **kwargs: Any) -> str: ...


# ----------------------------------------------------------------------
# Log level definition – provides ordering and associated metadata
# ----------------------------------------------------------------------
class LogLevel(Enum):
    DEBUG = auto()
    INFO = auto()
    WARN = auto()
    ERROR = auto()

    @property
    def label(self) -> str:
        """Human‑readable label used in the output string."""
        return f"{self.name}:"

    @property
    def colour(self) -> str:
        """Colour code from ``colorama``."""
        return cast(
            str,
            {
                LogLevel.DEBUG: Fore.BLUE,
                LogLevel.INFO: Fore.GREEN,
                LogLevel.WARN: Fore.YELLOW,
                LogLevel.ERROR: Fore.RED,
            }[self],
        )


# ----------------------------------------------------------------------
# Core logger class – pure, testable, and easily extensible
# ----------------------------------------------------------------------
class ColoredLogger:
    """
    Timestamped, colour‑coded logger.

    Example
    -------
    >>> log = ColoredLogger(level=\"debug\")
    >>> log.info(\"Server started on port %d\", 8080)
    2025-12-20 14:33:12 INFO: Server started on port 8080
    """

    _DEFAULT_FMT = "{time} {label} {msg}"
    _DEFAULT_TIME_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(
        self,
        level: str = "INFO",
        fmt: str = _DEFAULT_FMT,
        time_format: str = _DEFAULT_TIME_FMT,
        formatter: Optional[_Formatter] = None,
    ) -> None:
        # Normalise level string → Enum
        try:
            self._level: LogLevel = LogLevel[level.upper()]
        except KeyError as exc:
            raise ValueError(f"Invalid level: {level}") from exc

        self._fmt = fmt
        self._time_format = time_format
        # Default formatter uses ``str.format``; callers may inject any callable
        self._formatter: Callable[[str, Any], str] = (
            formatter if formatter is not None else lambda f, *a, **kw: f.format(*a, **kw)
        )

    # ------------------------------------------------------------------
    # Public API – one method per log level
    # ------------------------------------------------------------------
    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(LogLevel.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(LogLevel.INFO, msg, *args, **kwargs)

    def warn(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(LogLevel.WARN, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        self._log(LogLevel.ERROR, msg, *args, **kwargs)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------
    def set_level(self, level: str) -> None:
        """Change the minimum level that will be emitted."""
        try:
            self._level = LogLevel[level.upper()]
        except KeyError as exc:
            raise ValueError(f"Invalid level: {level}") from exc

    # ------------------------------------------------------------------
    # Internal helpers – kept private for testability
    # ------------------------------------------------------------------
    def _should_log(self, level: LogLevel) -> bool:
        """Return ``True`` if *level* is equal or higher than the current threshold."""
        return level.value >= self._level.value

    def _prepare_message(self, msg: str, *args: Any, **kwargs: Any) -> str:
        """
        Apply the formatter, falling back to ``%``‑formatting or simple concatenation
        if the primary formatter raises.
        """
        if not (args or kwargs):
            return msg

        try:
            return self._formatter(msg, *args, **kwargs)
        except Exception:
            # ``%`` formatting is a common legacy style
            try:
                return msg % args
            except Exception:
                # Final fallback – join everything as strings
                extras = " ".join(map(str, args))
                extras_kw = " ".join(f"{k}={v!r}" for k, v in kwargs.items())
                return " ".join(filter(None, (msg, extras, extras_kw)))

    def _format(self, level: LogLevel, message: str) -> str:
        """Inject timestamp, label and message into the user‑supplied format string."""
        now = datetime.datetime.now().strftime(self._time_format)
        return self._fmt.format(time=now, label=level.label, msg=message)

    def _output(self, level: LogLevel, formatted: str) -> None:
        """Write coloured output to the appropriate stream."""
        coloured = f"{level.colour}{formatted}{Style.RESET_ALL}"
        stream = sys.stderr if level is LogLevel.ERROR else sys.stdout
        print(coloured, file=stream)

    def _log(self, level: LogLevel, msg: str, *args: Any, **kwargs: Any) -> None:
        """Core logging pipeline – early‑exit if the level is filtered out."""
        if not self._should_log(level):
            return
        prepared = self._prepare_message(msg, *args, **kwargs)
        self._output(level, self._format(level, prepared))
