"""Visualization utilities for structural break analysis."""

from regimes.visualization.andrews_ploberger import plot_f_sequence
from regimes.visualization.breaks import (
    plot_break_confidence,
    plot_breaks,
    plot_regime_means,
)
from regimes.visualization.cusum import plot_cusum, plot_cusum_sq
from regimes.visualization.diagnostics import (
    plot_actual_fitted,
    plot_diagnostics,
    plot_residual_acf,
    plot_residual_distribution,
    plot_scaled_residuals,
)
from regimes.visualization.markov import (
    plot_ic,
    plot_parameter_time_series,
    plot_regime_shading,
    plot_smoothed_probabilities,
    plot_transition_matrix,
)
from regimes.visualization.params import plot_params_over_time
from regimes.visualization.rolling import plot_rolling_coefficients
from regimes.visualization.style import (
    REGIMES_COLOR_CYCLE,
    REGIMES_COLORS,
    add_break_dates,
    add_confidence_band,
    add_source,
    get_style,
    label_line_end,
    set_style,
    shade_regimes,
    use_style,
)

__all__ = [
    "REGIMES_COLORS",
    "REGIMES_COLOR_CYCLE",
    "add_break_dates",
    "add_confidence_band",
    "add_source",
    "get_style",
    "label_line_end",
    "plot_actual_fitted",
    "plot_break_confidence",
    "plot_breaks",
    "plot_cusum",
    "plot_cusum_sq",
    "plot_diagnostics",
    "plot_f_sequence",
    "plot_ic",
    "plot_parameter_time_series",
    "plot_params_over_time",
    "plot_regime_means",
    "plot_regime_shading",
    "plot_residual_acf",
    "plot_residual_distribution",
    "plot_rolling_coefficients",
    "plot_scaled_residuals",
    "plot_smoothed_probabilities",
    "plot_transition_matrix",
    "set_style",
    "shade_regimes",
    "use_style",
]
