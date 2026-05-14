"""
Color-aware terminal log formatter.

Level colors (file handlers automatically get plain text — no ANSI bleed):
  DEBUG    → dim grey
  INFO     → white, with content-aware tints:
               ✅ / PASS / PROMOTED / landed  → green
               ❌ / REJECTED / ERROR          → red
               ⚠️ / WARN / 🚫 / blocked       → yellow
               === / ─── (banners)            → bold blue
               📊 (metrics lines)             → cyan
               🌱 / 🧬 / 🔬 (lifecycle)       → bright cyan
  WARNING  → yellow
  ERROR    → red
  CRITICAL → bold red
"""

import logging
import re
import sys

_RESET = "\033[0m"
_BOLD  = "\033[1m"
_DIM   = "\033[2m"

_RED          = "\033[31m"
_GREEN        = "\033[32m"
_YELLOW       = "\033[33m"
_BLUE         = "\033[34m"
_CYAN         = "\033[36m"
_BRIGHT_RED   = "\033[91m"
_BRIGHT_GREEN = "\033[92m"
_BRIGHT_CYAN  = "\033[96m"
_BRIGHT_WHITE = "\033[97m"

_LEVEL_COLORS = {
    logging.DEBUG:    _DIM,
    logging.INFO:     "",            # determined per-message below
    logging.WARNING:  _YELLOW,
    logging.ERROR:    _RED,
    logging.CRITICAL: _BOLD + _BRIGHT_RED,
}

# Ordered list of (pattern, ansi) — first match wins for INFO lines
_INFO_TINTS = [
    # Banners / dividers
    (re.compile(r"[═=─]{6,}|ROUND\s+\d+|SUMMARY"), _BOLD + _BLUE),
    # Success
    (re.compile(r"✅|PROMOTED|landed(?!_but)|PASS|🌱|🏆"), _BRIGHT_GREEN),
    # Hard failure / error
    (re.compile(r"❌|REJECTED|ERROR|FAIL"), _BRIGHT_RED),
    # Soft warning / blocked
    (re.compile(r"⚠️|WARN|🚫|blocked|landed_but_correlated"), _YELLOW),
    # Metrics / IS result lines
    (re.compile(r"📊|Sharpe=|Fitness=|Turnover=|self_corr=|perf_gain="), _CYAN),
    # Lifecycle banners
    (re.compile(r"🧬|🔬|🧪|🔍|🌿"), _BRIGHT_CYAN),
]


def _tint_info(message: str) -> str:
    for pattern, color in _INFO_TINTS:
        if pattern.search(message):
            return color + message + _RESET
    return message


class ColorFormatter(logging.Formatter):
    """
    Formatter that adds ANSI color to terminal output.
    File handlers should use a plain Formatter; this one should only be
    attached to StreamHandlers pointed at a real TTY.
    """

    def __init__(self, fmt: str = "%(asctime)s [%(levelname)s] %(message)s",
                 datefmt: str = "%H:%M:%S"):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno, "")
        plain = super().format(record)

        if record.levelno == logging.INFO:
            # For INFO, tint by message content rather than uniform level color
            return _tint_info(plain)
        elif color:
            return color + plain + _RESET
        return plain


def apply_color_logging(
    level: int = logging.INFO,
    fmt: str = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt: str = "%H:%M:%S",
    log_file: str | None = None,
    noisy_loggers: list[str] | None = None,
) -> None:
    """
    Configure root logger with:
      - ColorFormatter on stderr (only when stderr is a real TTY)
      - Plain Formatter on log_file (when provided)

    Call this instead of logging.basicConfig().
    """
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    # Terminal handler
    stream_handler = logging.StreamHandler(sys.stderr)
    is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    if is_tty:
        stream_handler.setFormatter(ColorFormatter(fmt=fmt, datefmt=datefmt))
    else:
        stream_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(stream_handler)

    # Optional file handler — always plain text
    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(file_handler)

    # Silence noisy third-party loggers
    for name in (noisy_loggers or []):
        logging.getLogger(name).setLevel(logging.WARNING)
