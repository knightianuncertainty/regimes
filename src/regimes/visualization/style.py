"""Plotting style configuration for regimes.

This module provides a consistent visual identity for all regimes plots,
inspired by The Economist, the Financial Times, and Edward Tufte's principles:
maximize data-ink, minimize chart junk, and let the data speak.

See PLOTTING_STYLE.md (Version 1.0) for full specification.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from matplotlib.axes import Axes
    from numpy.typing import ArrayLike

# -----------------------------------------------------------------------------
# Color Palette
# -----------------------------------------------------------------------------

REGIMES_COLORS: dict[str, str] = {
    # Primary Economist-inspired cycle
    "blue": "#006BA2",  # Primary series, parameter estimates
    "red": "#DB444B",  # Secondary series, rejection regions, significance
    "teal": "#3EBCD2",  # Tertiary series
    "green": "#379A8B",  # Alternative series
    "gold": "#EBB434",  # Highlights, warnings
    "grey": "#758D99",  # Context lines, break dates, secondary elements
    "mauve": "#9A607F",  # Additional series if needed
    # Supplementary
    "light_grey": "#d4d4d4",  # Gridlines
    "near_black": "#333333",  # Text, axis lines
    "regime_tint_a": "#f0f4f8",  # Light blue-grey for odd regimes
    "regime_tint_b": "#ffffff",  # White for even regimes
}

# Color cycle as list for matplotlib prop_cycle
REGIMES_COLOR_CYCLE: list[str] = [
    REGIMES_COLORS["blue"],
    REGIMES_COLORS["red"],
    REGIMES_COLORS["teal"],
    REGIMES_COLORS["green"],
    REGIMES_COLORS["gold"],
    REGIMES_COLORS["grey"],
    REGIMES_COLORS["mauve"],
]


# -----------------------------------------------------------------------------
# Style Configuration
# -----------------------------------------------------------------------------


def get_style() -> dict[str, Any]:
    """Get the regimes rcParams style dictionary.

    Returns
    -------
    dict[str, Any]
        Dictionary of matplotlib rcParams for the regimes style.

    Notes
    -----
    This function lazily imports matplotlib to avoid import-time overhead.
    The style follows the specification in PLOTTING_STYLE.md (Version 1.0).
    """
    from cycler import cycler

    return {
        # Figure
        "figure.figsize": (10, 5),
        "figure.dpi": 150,
        "figure.facecolor": "white",
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        # Axes
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.8,
        "axes.edgecolor": REGIMES_COLORS["near_black"],
        "axes.labelsize": 11,
        "axes.labelcolor": REGIMES_COLORS["near_black"],
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlepad": 15,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "axes.prop_cycle": cycler("color", REGIMES_COLOR_CYCLE),
        # Grid
        "grid.color": REGIMES_COLORS["light_grey"],
        "grid.linewidth": 0.5,
        "grid.alpha": 0.7,
        # Lines
        "lines.linewidth": 2.0,
        "lines.antialiased": True,
        # Ticks
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.color": REGIMES_COLORS["near_black"],
        "ytick.color": REGIMES_COLORS["near_black"],
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 8,
        "ytick.major.pad": 8,
        # Legend
        "legend.frameon": False,
        "legend.fontsize": 10,
        # Saving
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    }


def set_style() -> None:
    """Apply the regimes plotting style globally.

    This modifies matplotlib's global rcParams. Use `use_style()` as a
    context manager for temporary style application.

    Examples
    --------
    >>> from regimes.visualization.style import set_style
    >>> set_style()  # All subsequent plots use regimes style
    """
    import matplotlib as mpl

    # Any: newer matplotlib stubs type the keys of rcParams as literals
    style: Any = get_style()
    mpl.rcParams.update(style)


@contextmanager
def use_style() -> Iterator[None]:
    """Context manager for temporary style application.

    Applies the regimes style within the context and restores the
    original rcParams on exit. This is the recommended approach for
    internal use within plotting functions.

    Yields
    ------
    None

    Examples
    --------
    >>> from regimes.visualization.style import use_style
    >>> with use_style():
    ...     # Plotting code here uses regimes style
    ...     pass
    >>> # Original rcParams are restored
    """
    import matplotlib.pyplot as plt

    with plt.style.context(get_style()):
        yield


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def label_line_end(
    ax: Axes,
    x: ArrayLike,
    y: ArrayLike,
    label: str,
    color: str,
    offset: tuple[float, float] = (8, 0),
    fontsize: float = 10,
    fontweight: str = "bold",
) -> None:
    """Label a line at its last data point (replaces legend).

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to annotate.
    x : ArrayLike
        X-coordinates of the line.
    y : ArrayLike
        Y-coordinates of the line.
    label : str
        Text label to place at the end of the line.
    color : str
        Color for the label text.
    offset : tuple[float, float]
        Offset in points from the endpoint (x_offset, y_offset).
    fontsize : float
        Font size for the label.
    fontweight : str
        Font weight for the label.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from regimes.visualization.style import label_line_end
    >>> fig, ax = plt.subplots()
    >>> x = [0, 1, 2, 3]
    >>> y = [1, 2, 3, 4]
    >>> ax.plot(x, y, color='#006BA2')
    >>> label_line_end(ax, x, y, 'Series A', '#006BA2')
    """
    import numpy as np

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    ax.annotate(
        label,
        xy=(x_arr[-1], y_arr[-1]),
        xytext=offset,
        textcoords="offset points",
        fontsize=fontsize,
        color=color,
        va="center",
        fontweight=fontweight,
        clip_on=False,
    )


def add_break_dates(
    ax: Axes,
    break_dates: Sequence[int | float],
    color: str | None = None,
    linestyle: str = "--",
    linewidth: float = 0.8,
    alpha: float = 0.7,
    labels: Sequence[str] | None = None,
    zorder: int = 1,
) -> None:
    """Add vertical dashed lines at structural break dates.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    break_dates : Sequence[int | float]
        Break point locations (x-coordinates).
    color : str | None
        Color for break lines. If None, uses palette grey.
    linestyle : str
        Line style for break lines.
    linewidth : float
        Line width for break lines.
    alpha : float
        Alpha (transparency) for break lines.
    labels : Sequence[str] | None
        Optional labels to place at the top of each break line.
    zorder : int
        Z-order for the break lines.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from regimes.visualization.style import add_break_dates
    >>> fig, ax = plt.subplots()
    >>> ax.plot([0, 100], [0, 1])
    >>> add_break_dates(ax, [25, 50, 75])
    """
    if color is None:
        color = REGIMES_COLORS["grey"]

    for i, bd in enumerate(break_dates):
        ax.axvline(
            x=bd,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
        if labels and i < len(labels):
            ax.text(
                bd,
                ax.get_ylim()[1],
                f" {labels[i]}",
                fontsize=9,
                color=color,
                alpha=alpha,
                va="top",
                ha="left",
                rotation=0,
            )


def add_confidence_band(
    ax: Axes,
    x: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    color: str | None = None,
    alpha: float = 0.15,
    zorder: int = 0,
) -> None:
    """Add a translucent confidence interval band.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    x : ArrayLike
        X-coordinates.
    lower : ArrayLike
        Lower bound of the confidence interval.
    upper : ArrayLike
        Upper bound of the confidence interval.
    color : str | None
        Color for the band. If None, uses palette blue.
    alpha : float
        Alpha (transparency) for the band.
    zorder : int
        Z-order for the band.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from regimes.visualization.style import add_confidence_band
    >>> fig, ax = plt.subplots()
    >>> x = np.arange(100)
    >>> y = np.sin(x / 10)
    >>> ax.plot(x, y)
    >>> add_confidence_band(ax, x, y - 0.2, y + 0.2)
    """
    if color is None:
        color = REGIMES_COLORS["blue"]

    ax.fill_between(
        x,
        lower,
        upper,
        color=color,
        alpha=alpha,
        linewidth=0,
        zorder=zorder,
    )


def shade_regimes(
    ax: Axes,
    break_dates: Sequence[int | float],
    start: int | float,
    end: int | float,
    colors: tuple[str, str] | None = None,
    alpha: float = 0.5,
    zorder: int = 0,
) -> None:
    """Shade alternating regimes with barely-visible tints.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to draw on.
    break_dates : Sequence[int | float]
        Break point locations (x-coordinates).
    start : int | float
        Start of the sample period.
    end : int | float
        End of the sample period.
    colors : tuple[str, str] | None
        Two colors for alternating regimes. If None, uses palette tints.
    alpha : float
        Alpha (transparency) for the shading.
    zorder : int
        Z-order for the shading.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from regimes.visualization.style import shade_regimes
    >>> fig, ax = plt.subplots()
    >>> ax.plot([0, 100], [0, 1])
    >>> shade_regimes(ax, [30, 60], 0, 100)
    """
    if colors is None:
        colors = (
            REGIMES_COLORS["regime_tint_a"],
            REGIMES_COLORS["regime_tint_b"],
        )

    boundaries = [start] + list(break_dates) + [end]
    for i in range(len(boundaries) - 1):
        ax.axvspan(
            boundaries[i],
            boundaries[i + 1],
            facecolor=colors[i % len(colors)],
            alpha=alpha,
            zorder=zorder,
        )


def add_source(
    ax: Axes,
    text: str,
    fontsize: float = 8,
    color: str = "#999999",
) -> None:
    """Add a source/note line at the bottom-left of the plot.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to annotate.
    text : str
        Source text to display.
    fontsize : float
        Font size for the text.
    color : str
        Color for the text.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from regimes.visualization.style import add_source
    >>> fig, ax = plt.subplots()
    >>> ax.plot([0, 1], [0, 1])
    >>> add_source(ax, "Source: Federal Reserve")
    """
    ax.text(
        0,
        -0.12,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        color=color,
        va="top",
        ha="left",
    )
