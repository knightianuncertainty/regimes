"""Warnings that point at the caller's code."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

_PACKAGE = str(Path(__file__).resolve().parent) + os.sep


def warn_user(message: str, category: type[Warning] = UserWarning) -> None:
    """Issue a warning attributed to the first frame outside this package.

    A fixed ``stacklevel`` points at the right line only for one call path;
    the same warning is reached through several (a model method, a
    convenience method of another class), so the frames are walked instead.
    """
    level = 2
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename.startswith(_PACKAGE):
        frame = frame.f_back  # type: ignore[assignment]
        level += 1
    warnings.warn(message, category, stacklevel=level)
