"""
cv_utils.py — Shared Computer Vision Utilities
================================================
Visualization helpers used across all CV track modules.

Usage:
    import sys
    sys.path.insert(0, '../../shared')
    from cv_utils import show_tensor, plot_feature_maps, overlay_mask, plot_channels
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import torch
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
from PIL import Image


# ─────────────────────────────────────────────────────────────
#  1. TENSOR → IMAGE DISPLAY
# ─────────────────────────────────────────────────────────────

def show_tensor(
    tensor: torch.Tensor,
    title: str = "",
    denormalize: bool = False,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (5, 5),
) -> None:
    """
    Display a CHW or HWC image tensor using matplotlib.

    Args:
        tensor:      Image tensor. Shape: [C, H, W] or [H, W, C] or [H, W] (grayscale).
        title:       Title for the plot.
        denormalize: If True, reverse ImageNet normalization before display.
        mean:        Mean used for normalization (only used if denormalize=True).
        std:         Std used for normalization (only used if denormalize=True).
        ax:          Matplotlib Axes to draw on. Creates a new figure if None.
        figsize:     Figure size (only used if ax is None).
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    img = tensor.detach().cpu().float()

    # Convert CHW → HWC
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = img.permute(1, 2, 0)

    # Denormalize (ImageNet stats by default)
    if denormalize and img.ndim == 3 and img.shape[2] == 3:
        _mean = torch.tensor(mean, dtype=img.dtype)
        _std = torch.tensor(std, dtype=img.dtype)
        img = img * _std + _mean

    # Squeeze single channel
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(2)

    # Clip to [0, 1]
    img = img.clamp(0.0, 1.0).numpy()

    cmap = "gray" if img.ndim == 2 else None
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=11, pad=6)
    ax.axis("off")


# ─────────────────────────────────────────────────────────────
#  2. FEATURE MAP GRID
# ─────────────────────────────────────────────────────────────

def plot_feature_maps(
    feature_maps: torch.Tensor,
    max_channels: int = 16,
    title: str = "Feature Maps",
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "viridis",
) -> None:
    """
    Visualise CNN feature maps as a grid.

    Args:
        feature_maps: Tensor of shape [1, C, H, W] or [C, H, W].
        max_channels:  Maximum number of channels to display.
        title:         Overall figure title.
        figsize:       Figure size; auto-computed if None.
        cmap:          Colormap for the heatmaps.
    """
    fmaps = feature_maps.detach().cpu().float()

    # Remove batch dim if present
    if fmaps.ndim == 4:
        fmaps = fmaps[0]

    n_channels = min(fmaps.shape[0], max_channels)
    cols = min(8, n_channels)
    rows = (n_channels + cols - 1) // cols

    if figsize is None:
        figsize = (cols * 2, rows * 2 + 0.6)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(title, fontsize=13, y=1.01)

    axes_flat = np.array(axes).flatten() if n_channels > 1 else [axes]

    for i, ax in enumerate(axes_flat):
        if i < n_channels:
            fm = fmaps[i].numpy()
            ax.imshow(fm, cmap=cmap)
            ax.set_title(f"ch {i}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
#  3. RGB CHANNEL DECOMPOSITION
# ─────────────────────────────────────────────────────────────

def plot_channels(
    tensor: torch.Tensor,
    figsize: Tuple[int, int] = (14, 3),
    denormalize: bool = False,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> None:
    """
    Plot an RGB image alongside its individual R, G, B channel heatmaps.

    Args:
        tensor:      CHW image tensor.
        figsize:     Figure size.
        denormalize: Reverse normalization before display.
    """
    img = tensor.detach().cpu().float()

    if denormalize:
        _mean = torch.tensor(mean).view(3, 1, 1)
        _std = torch.tensor(std).view(3, 1, 1)
        img = img * _std + _mean

    img = img.clamp(0, 1)
    hwc = img.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    channel_names = ["RGB (original)", "Red channel", "Green channel", "Blue channel"]
    cmaps = [None, "Reds_r", "Greens_r", "Blues_r"]

    axes[0].imshow(hwc)
    axes[0].set_title(channel_names[0])
    axes[0].axis("off")

    for i in range(3):
        axes[i + 1].imshow(img[i].numpy(), cmap=cmaps[i + 1], vmin=0, vmax=1)
        axes[i + 1].set_title(channel_names[i + 1])
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
#  4. SEGMENTATION MASK OVERLAY
# ─────────────────────────────────────────────────────────────

def overlay_mask(
    image: torch.Tensor,
    mask: torch.Tensor,
    alpha: float = 0.45,
    color: Tuple[float, float, float] = (1.0, 0.2, 0.2),
    title: str = "",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (5, 5),
) -> None:
    """
    Overlay a binary segmentation mask on top of an image.

    Args:
        image:   CHW image tensor, values in [0, 1].
        mask:    HW binary mask tensor (0 = background, 1 = foreground), or [1, H, W].
        alpha:   Opacity of the overlay color.
        color:   RGB tuple for the mask overlay color.
        title:   Plot title.
        ax:      Matplotlib Axes; creates new figure if None.
        figsize: Figure size.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    img = image.detach().cpu().float().clamp(0, 1)
    if img.ndim == 3:
        img = img.permute(1, 2, 0).numpy()
    else:
        img = img.numpy()

    msk = mask.detach().cpu().float()
    if msk.ndim == 3:
        msk = msk.squeeze(0)
    msk = msk.numpy()

    overlay = img.copy()
    fg = msk > 0.5
    for c, v in enumerate(color):
        overlay[:, :, c] = np.where(fg, img[:, :, c] * (1 - alpha) + v * alpha, img[:, :, c])

    ax.imshow(overlay)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


# ─────────────────────────────────────────────────────────────
#  5. TRAINING CURVES
# ─────────────────────────────────────────────────────────────

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    title: str = "Training Curves",
    figsize: Tuple[int, int] = (12, 4),
) -> None:
    """
    Plot loss (and optionally accuracy) curves for train vs validation.

    Args:
        train_losses: List of training loss values per epoch.
        val_losses:   List of validation loss values per epoch.
        train_accs:   Optional list of training accuracy values per epoch.
        val_accs:     Optional list of validation accuracy values per epoch.
        title:        Figure title.
        figsize:      Figure size.
    """
    has_acc = train_accs is not None and val_accs is not None
    n_plots = 2 if has_acc else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    fig.suptitle(title, fontsize=13)

    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_losses) + 1)

    # Loss
    axes[0].plot(epochs, train_losses, "b-o", markersize=4, label="Train Loss")
    axes[0].plot(epochs, val_losses, "r-o", markersize=4, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    if has_acc:
        axes[1].plot(epochs, train_accs, "b-o", markersize=4, label="Train Acc")
        axes[1].plot(epochs, val_accs, "r-o", markersize=4, label="Val Acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy (%)")
        axes[1].set_title("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
#  6. CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "Blues",
    normalize: bool = True,
) -> None:
    """
    Plot a confusion matrix heatmap.

    Args:
        cm:           2D numpy array, shape [n_classes, n_classes].
        class_names:  List of class label strings.
        title:        Plot title.
        figsize:      Figure size; auto-computed if None.
        cmap:         Colormap.
        normalize:    Normalise rows to [0, 1] for percentage display.
    """
    n = len(class_names)
    if figsize is None:
        figsize = (max(6, n), max(5, n - 1))

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_display = cm.astype(float) / row_sums
    else:
        cm_display = cm.astype(float)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap, vmin=0, vmax=1 if normalize else None)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    thresh = cm_display.max() / 2.0
    for i in range(n):
        for j in range(n):
            val = f"{cm_display[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if cm_display[i, j] > thresh else "black", fontsize=8)

    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
#  7. KERNEL / WEIGHT VISUALISATION
# ─────────────────────────────────────────────────────────────

def visualize_conv_kernels(
    weight: torch.Tensor,
    max_kernels: int = 16,
    title: str = "Conv Kernels",
    figsize: Optional[Tuple[int, int]] = None,
) -> None:
    """
    Visualise the learned kernels of a Conv2d weight tensor.

    Args:
        weight:      Conv2d weight tensor: [out_channels, in_channels, kH, kW].
        max_kernels: Maximum number of kernels to display.
        title:       Figure title.
        figsize:     Figure size.
    """
    w = weight.detach().cpu().float()
    n = min(w.shape[0], max_kernels)

    # For multi-channel kernels: use the mean across input channels
    if w.shape[1] > 1:
        kernels = w[:n].mean(dim=1)  # [n, kH, kW]
    else:
        kernels = w[:n, 0]  # [n, kH, kW]

    cols = min(8, n)
    rows = (n + cols - 1) // cols
    if figsize is None:
        figsize = (cols * 1.5, rows * 1.5 + 0.5)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.suptitle(title, fontsize=12)
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for i, ax in enumerate(axes_flat):
        if i < n:
            k = kernels[i].numpy()
            # Normalize per-kernel for visibility
            k = (k - k.min()) / (k.max() - k.min() + 1e-8)
            ax.imshow(k, cmap="RdBu_r", vmin=0, vmax=1)
            ax.set_title(f"K{i}", fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
#  8. DEVICE HELPER
# ─────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_model_summary(model: torch.nn.Module, input_size: Tuple[int, ...] = (1, 3, 224, 224)) -> None:
    """
    Print parameter count and estimated model size.

    Args:
        model:      PyTorch model.
        input_size: Input tensor shape (batch, C, H, W).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total * 4 / (1024 ** 2)  # float32 = 4 bytes
    print(f"{'─'*40}")
    print(f"  Total parameters   : {total:>12,}")
    print(f"  Trainable params   : {trainable:>12,}")
    print(f"  Non-trainable      : {total - trainable:>12,}")
    print(f"  Est. size (fp32)   : {size_mb:>10.2f} MB")
    print(f"{'─'*40}")
