import logging
import sys


_FORMAT = "[%(asctime)s] %(levelname)s %(name)s — %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"
_initialized = False

_PROJECT = frozenset(("__main__", "brain", "memory", "bot", "config", "logger"))


class _ProjectFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        top = record.name.split(".", 1)[0]
        return top in _PROJECT or record.levelno >= logging.WARNING


def init(level: int = logging.INFO) -> None:
    global _initialized
    if _initialized:
        return
    _initialized = True

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    handler.addFilter(_ProjectFilter())

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)

    for entry in logging.Logger.manager.loggerDict.values():
        if isinstance(entry, logging.Logger) and not entry.propagate:
            if entry.name.split(".", 1)[0] not in _PROJECT:
                entry.handlers.clear()
                entry.propagate = True


def get(name: str) -> logging.Logger:
    return logging.getLogger(name)
