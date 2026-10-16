"""shared/__init__.py"""
from .plotters import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_feature_importance,
    plot_learning_curves,
    plot_decision_boundary,
    plot_residuals,
    print_classification_summary,
    compare_models_bar,
    PALETTE,
    MODEL_COLORS,
)

__all__ = [
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_precision_recall_curves",
    "plot_feature_importance",
    "plot_learning_curves",
    "plot_decision_boundary",
    "plot_residuals",
    "print_classification_summary",
    "compare_models_bar",
    "PALETTE",
    "MODEL_COLORS",
]
