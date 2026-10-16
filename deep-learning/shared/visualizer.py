"""
Utility visualizers for Deep Learning Lab
Provides clean decision boundary and loss curve plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """
    Plots the decision boundary created by a PyTorch model on 2D data.
    
    Args:
        model: PyTorch nn.Module taking 2D tensor input and outputting logits/probabilities
        X: torch.Tensor or numpy.ndarray of shape [N, 2]
        y: torch.Tensor or numpy.ndarray of shape [N]
        title: Title of the plot
    """
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().numpy()
    else:
        X_np = X
        
    if isinstance(y, torch.Tensor):
        y_np = y.detach().cpu().numpy()
    else:
        y_np = y

    # Define grid boundaries
    x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5
    h = 0.02
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

    # Predict over grid
    model.eval()
    device = next(model.parameters()).device
    grid_tensor = grid_tensor.to(device)
    
    with torch.no_grad():
        preds = model(grid_tensor)
        # Handle binary or multiclass
        if preds.shape[-1] == 1:
            probs = torch.sigmoid(preds)
            preds_class = (probs > 0.5).int().cpu().numpy()
        else:
            preds_class = torch.argmax(preds, dim=-1).cpu().numpy()
            
    Z = preds_class.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
    scatter = plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np, s=40, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title(title, fontsize=13, fontweight='bold')
    plt.xlabel("Feature 1 ($x_1$)")
    plt.ylabel("Feature 2 ($x_2$)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
