"""Warnings that point at the caller's code."""

from __future__ import annotations

import os
import sys
import warnings

# the folder of this package as the interpreter names it in its frames (no
# resolving of links, which would differ from the path the package was
# imported by), compared without regard to case where the file system is so
_PACKAGE = os.path.normcase(os.path.dirname(os.path.abspath(__file__))) + os.sep  # noqa: PTH100, PTH120


def _inside_package(filename: str) -> bool:
    path = os.path.normcase(os.path.abspath(filename))  # noqa: PTH100
    return path.startswith(_PACKAGE)


def warn_user(message: str, category: type[Warning] = UserWarning) -> None:
    """Issue a warning attributed to the first frame outside this package.

    A fixed ``stacklevel`` points at the right line only for one call path;
    the same warning is reached through several (a model method, a
    convenience method of another class), so the frames are walked instead.
    """
    level = 2
    frame = sys._getframe(1)
    while frame is not None and _inside_package(frame.f_code.co_filename):
        frame = frame.f_back  # type: ignore[assignment]
        level += 1
    warnings.warn(message, category, stacklevel=level)
