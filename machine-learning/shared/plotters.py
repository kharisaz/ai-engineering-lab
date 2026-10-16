"""
shared/plotters.py
==================
Utility functions for ML visualization used across all modules.
Provides consistent, publication-quality confusion matrix and ROC curve plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Color palette consistent across all modules
# ─────────────────────────────────────────────
PALETTE = {
    "primary": "#3b82f6",
    "secondary": "#8b5cf6",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "neutral": "#6b7280",
    "background": "#f8fafc",
}

MODEL_COLORS = [
    "#3b82f6", "#10b981", "#f59e0b", "#ef4444",
    "#8b5cf6", "#06b6d4", "#f97316", "#84cc16",
]


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels=None,
    title="Confusion Matrix",
    figsize=(7, 6),
    normalize=None,
    colormap="Blues",
    ax=None,
):
    """
    Plot a styled confusion matrix.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : list, optional
        Class label names.
    title : str
        Plot title.
    figsize : tuple
        Figure size (ignored if ax is provided).
    normalize : {'true', 'pred', 'all', None}
        Normalization method. None = raw counts.
    colormap : str
        Matplotlib colormap name.
    ax : matplotlib Axes, optional
        If provided, plot into this axes.

    Returns
    -------
    fig, ax
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(
        ax=ax,
        colorbar=True,
        cmap=colormap,
        values_format=".2f" if normalize else "d",
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    if normalize:
        ax.set_title(f"{title}\n(normalized by '{normalize}')", fontsize=13, fontweight="bold")

    fig.tight_layout()
    return fig, ax


def plot_roc_curves(
    y_true,
    y_scores_dict,
    title="ROC Curves",
    figsize=(8, 6),
    ax=None,
):
    """
    Plot ROC curves for multiple classifiers on a single axes.

    Parameters
    ----------
    y_true : array-like
        Binary true labels (0/1).
    y_scores_dict : dict
        Mapping of {model_name: predicted_probabilities}.
        Probabilities should be for the positive class.
    title : str
        Plot title.
    figsize : tuple
        Figure size (ignored if ax is provided).
    ax : matplotlib Axes, optional
        If provided, plot into this axes.

    Returns
    -------
    fig, ax, dict of {model_name: auc_score}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    auc_scores = {}

    for i, (model_name, y_score) in enumerate(y_scores_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        auc_scores[model_name] = roc_auc
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.plot(fpr, tpr, color=color, lw=2.5,
                label=f"{model_name} (AUC = {roc_auc:.3f})")

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(PALETTE["background"])

    fig.tight_layout()
    return fig, ax, auc_scores


def plot_precision_recall_curves(
    y_true,
    y_scores_dict,
    title="Precision-Recall Curves",
    figsize=(8, 6),
    ax=None,
):
    """
    Plot Precision-Recall curves for multiple classifiers.

    Parameters
    ----------
    y_true : array-like
        Binary true labels (0/1).
    y_scores_dict : dict
        Mapping of {model_name: predicted_probabilities}.
    title : str
        Plot title.
    figsize : tuple
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax, dict of {model_name: avg_precision_score}
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ap_scores = {}
    baseline = np.mean(y_true)

    for i, (model_name, y_score) in enumerate(y_scores_dict.items()):
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ap_scores[model_name] = ap
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.plot(recall, precision, color=color, lw=2.5,
                label=f"{model_name} (AP = {ap:.3f})")

    ax.axhline(y=baseline, color="k", linestyle="--", lw=1.5,
               label=f"Random Classifier (AP = {baseline:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(PALETTE["background"])

    fig.tight_layout()
    return fig, ax, ap_scores


def plot_feature_importance(
    feature_names,
    importances,
    title="Feature Importances",
    top_n=20,
    figsize=(10, 7),
    color=None,
    ax=None,
):
    """
    Plot a horizontal bar chart of feature importances.

    Parameters
    ----------
    feature_names : list of str
        Names of features.
    importances : array-like
        Importance scores (higher = more important).
    title : str
        Plot title.
    top_n : int
        Number of top features to display.
    figsize : tuple
    color : str, optional
        Bar color. Defaults to primary palette color.
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    if color is None:
        color = PALETTE["primary"]

    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    # Reverse for horizontal bar (highest at top)
    top_features = top_features[::-1]
    top_importances = top_importances[::-1]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    bars = ax.barh(top_features, top_importances, color=color, alpha=0.85, edgecolor="white")

    # Value labels on bars
    for bar, val in zip(bars, top_importances):
        ax.text(
            bar.get_width() + max(top_importances) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_facecolor(PALETTE["background"])
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig, ax


def plot_learning_curves(
    train_scores,
    val_scores,
    train_sizes=None,
    title="Learning Curves",
    metric_name="Score",
    figsize=(9, 5),
    ax=None,
):
    """
    Plot training and validation learning curves.

    Parameters
    ----------
    train_scores : array-like, shape (n_splits,) or (n_train_sizes, n_splits)
        Training scores. If 2D, mean/std across splits is computed.
    val_scores : array-like
        Validation scores. Same shape as train_scores.
    train_sizes : array-like, optional
        X-axis values (number of training samples).
    title : str
    metric_name : str
        Name of the metric being plotted.
    figsize : tuple
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    train_scores = np.array(train_scores)
    val_scores = np.array(val_scores)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if train_sizes is None:
        train_sizes = np.arange(1, len(train_scores) + 1)

    if train_scores.ndim == 2:
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)
    else:
        train_mean = train_scores
        train_std = np.zeros_like(train_scores)
        val_mean = val_scores
        val_std = np.zeros_like(val_scores)

    ax.plot(train_sizes, train_mean, "o-", color=PALETTE["primary"],
            lw=2.5, label=f"Training {metric_name}")
    ax.fill_between(train_sizes,
                    train_mean - train_std,
                    train_mean + train_std,
                    alpha=0.15, color=PALETTE["primary"])

    ax.plot(train_sizes, val_mean, "s-", color=PALETTE["danger"],
            lw=2.5, label=f"Validation {metric_name}")
    ax.fill_between(train_sizes,
                    val_mean - val_std,
                    val_mean + val_std,
                    alpha=0.15, color=PALETTE["danger"])

    ax.set_xlabel("Training Examples", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor(PALETTE["background"])

    fig.tight_layout()
    return fig, ax


def plot_decision_boundary(
    model,
    X,
    y,
    feature_names=None,
    title="Decision Boundary",
    figsize=(8, 6),
    resolution=300,
    ax=None,
):
    """
    Plot the 2D decision boundary of a trained classifier.
    Works with any sklearn-compatible model.

    Parameters
    ----------
    model : sklearn estimator
        Trained classifier with a predict() method.
    X : array-like, shape (n_samples, 2)
        2D feature matrix.
    y : array-like, shape (n_samples,)
        True labels.
    feature_names : list of 2 strings, optional
        Names of the 2 features.
    title : str
    figsize : tuple
    resolution : int
        Grid resolution for the decision boundary mesh.
    ax : matplotlib Axes, optional

    Returns
    -------
    fig, ax
    """
    X = np.array(X)
    y = np.array(y)
    classes = np.unique(y)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    colors_bg = ["#bfdbfe", "#bbf7d0", "#fde68a", "#fecaca"]
    colors_pt = [PALETTE["primary"], PALETTE["success"], PALETTE["warning"], PALETTE["danger"]]
    markers = ["o", "s", "^", "D"]

    for i, cls in enumerate(classes):
        # Background region
        ax.contourf(
            xx, yy,
            (Z == cls).astype(int),
            levels=[0.5, 1.5],
            colors=[colors_bg[i % len(colors_bg)]],
            alpha=0.5,
        )
        # Data points
        mask = y == cls
        ax.scatter(
            X[mask, 0], X[mask, 1],
            c=colors_pt[i % len(colors_pt)],
            marker=markers[i % len(markers)],
            edgecolors="white",
            linewidths=0.5,
            s=60,
            label=f"Class {cls}",
            zorder=3,
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    if feature_names:
        ax.set_xlabel(feature_names[0], fontsize=11)
        ax.set_ylabel(feature_names[1], fontsize=11)
    ax.legend(fontsize=10)
    ax.set_facecolor(PALETTE["background"])
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    return fig, ax


def plot_residuals(
    y_true,
    y_pred,
    title="Residual Analysis",
    figsize=(13, 5),
):
    """
    Plot residual analysis for regression models (2 subplots).

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted values.
    title : str
    figsize : tuple

    Returns
    -------
    fig, axes (tuple of 2 axes)
    """
    residuals = np.array(y_true) - np.array(y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Residuals vs Predicted
    ax1.scatter(y_pred, residuals, alpha=0.5, color=PALETTE["primary"], edgecolors="white", s=40)
    ax1.axhline(y=0, color=PALETTE["danger"], linestyle="--", lw=2)
    ax1.set_xlabel("Predicted Values", fontsize=11)
    ax1.set_ylabel("Residuals", fontsize=11)
    ax1.set_title("Residuals vs Predicted", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor(PALETTE["background"])

    # Residual distribution
    ax2.hist(residuals, bins=40, color=PALETTE["secondary"], alpha=0.8, edgecolor="white")
    ax2.axvline(x=0, color=PALETTE["danger"], linestyle="--", lw=2, label="Zero Error")
    ax2.set_xlabel("Residual Value", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.set_title("Residual Distribution", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor(PALETTE["background"])

    fig.tight_layout()
    return fig, (ax1, ax2)


def print_classification_summary(y_true, y_pred, labels=None, title="Classification Report"):
    """
    Print a formatted classification report with header.

    Parameters
    ----------
    y_true : array-like
    y_pred : array-like
    labels : list, optional
        Target label names.
    title : str
    """
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))


def compare_models_bar(
    model_names,
    metric_values_dict,
    title="Model Comparison",
    figsize=(11, 5),
):
    """
    Create a grouped bar chart to compare multiple models across multiple metrics.

    Parameters
    ----------
    model_names : list of str
        Names of models being compared.
    metric_values_dict : dict
        {metric_name: list_of_values_per_model}
        e.g. {"Accuracy": [0.95, 0.92], "F1": [0.88, 0.85]}
    title : str
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    metrics = list(metric_values_dict.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=figsize)

    for i, model_name in enumerate(model_names):
        values = [metric_values_dict[m][i] for m in metrics]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, values, width,
            label=model_name,
            color=MODEL_COLORS[i % len(MODEL_COLORS)],
            alpha=0.87,
            edgecolor="white",
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=8.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_ylim(0, 1.12)
    ax.set_facecolor(PALETTE["background"])
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig, ax
