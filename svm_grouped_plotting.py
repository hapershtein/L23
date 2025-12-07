#!/usr/bin/env python3
"""
Visualization functions for grouped SVM classification.
Creates plots for decision boundaries, weight vectors, and hyperplane parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from svm_grouped_model import calculate_hyperplane_parameters


def plot_decision_boundary_with_hyperplane(X, y, model, scaler, feature_names, class_names,
                                           original_classes, filename='svm_visualization_grouped.png'):
    """Plot decision boundary, hyperplane, support vectors, and weight vector"""
    # Scale the data
    X_scaled = scaler.transform(X)

    # Get hyperplane parameters
    w, b, margin, w_norm = calculate_hyperplane_parameters(model)

    # Create a mesh to plot decision boundary
    h = 0.02
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict for each point in the mesh
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create the plot
    plt.figure(figsize=(16, 6))

    # Plot 1: Decision boundary with hyperplane and margins (scaled)
    plt.subplot(1, 3, 1)
    _plot_scaled_decision_boundary(xx, yy, Z, X_scaled, y, model, w,
                                   class_names, feature_names, x_min, x_max, y_min, y_max)

    # Plot 2: Original data with original class labels
    plt.subplot(1, 3, 2)
    _plot_original_classes(X, model, original_classes, feature_names)

    # Plot 3: Weight vector components and hyperplane parameters
    plt.subplot(1, 3, 3)
    _plot_hyperplane_parameters(w, b, margin, w_norm, model, feature_names)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as '{filename}'")
    plt.close()


def _plot_scaled_decision_boundary(xx, yy, Z, X_scaled, y, model, w,
                                   class_names, feature_names, x_min, x_max, y_min, y_max):
    """Helper function to plot scaled decision boundary"""
    # Plot decision boundary (filled contours)
    plt.contourf(xx, yy, Z > 0, alpha=0.3, levels=1, cmap=plt.cm.RdYlBu)

    # Plot hyperplane (decision boundary) and margins
    plt.contour(xx, yy, Z, colors=['k', 'gray', 'gray'],
                levels=[-1, 0, 1], linestyles=['dashed', 'solid', 'dashed'],
                linewidths=[2, 3, 2])

    # Plot data points
    colors = ['red', 'blue']
    for idx, class_name in enumerate(class_names):
        mask = y == idx
        plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                   c=colors[idx], label=class_name,
                   edgecolors='k', s=80, alpha=0.7)

    # Plot support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=200, linewidth=2, facecolors='none', edgecolors='green',
                label='Support Vectors')

    # Plot weight vector from origin
    scale_factor = 2.0
    origin_x = (x_min + x_max) / 2
    origin_y = (y_min + y_max) / 2
    plt.arrow(origin_x, origin_y, w[0] * scale_factor, w[1] * scale_factor,
              head_width=0.15, head_length=0.15, fc='purple', ec='purple',
              linewidth=3, label=f'Weight Vector w', zorder=5)

    plt.xlabel(f'{feature_names[0]} (scaled)', fontsize=11)
    plt.ylabel(f'{feature_names[1]} (scaled)', fontsize=11)
    plt.title('SVM Hyperplane and Weight Vector', fontsize=13, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)


def _plot_original_classes(X, model, original_classes, feature_names):
    """Helper function to plot original three classes"""
    original_y = np.zeros(len(X))
    original_y[:50] = 0   # Setosa
    original_y[50:100] = 1  # Versicolor
    original_y[100:] = 2  # Virginica

    colors_original = ['red', 'blue', 'green']
    for idx, class_name in enumerate(original_classes):
        mask = original_y == idx
        plt.scatter(X[mask, 0], X[mask, 1],
                   c=colors_original[idx], label=class_name,
                   edgecolors='k', s=80, alpha=0.7)

    # Mark support vectors
    support_vector_indices = model.support_
    plt.scatter(X[support_vector_indices, 0], X[support_vector_indices, 1],
                s=200, linewidth=2, facecolors='none', edgecolors='black',
                label=f'Support Vectors (n={len(support_vector_indices)})')

    plt.xlabel(feature_names[0], fontsize=11)
    plt.ylabel(feature_names[1], fontsize=11)
    plt.title('Original 3 Classes with Support Vectors', fontsize=13, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)


def _plot_hyperplane_parameters(w, b, margin, w_norm, model, feature_names):
    """Helper function to plot hyperplane parameters"""
    ax = plt.gca()
    ax.axis('off')

    # Title
    plt.text(0.5, 0.95, 'Hyperplane Parameters',
             fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)

    # Weight vector visualization
    plt.text(0.5, 0.85, 'Weight Vector (w)',
             fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)

    # Draw weight vector components
    _draw_weight_bars(w, ax, feature_names)

    # Parameters text
    params_text = f"""
Hyperplane Equation:
{w[0]:.4f} × x₁ + {w[1]:.4f} × x₂ + {b:.4f} = 0

||w|| = {w_norm:.4f}
Margin = 2/||w|| = {margin:.4f}

Support Vectors: {len(model.support_vectors_)}
    """

    plt.text(0.5, 0.35, params_text,
             fontsize=10, ha='center', va='top',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             transform=ax.transAxes)

    # Add interpretation
    interpretation = """
The weight vector w is perpendicular
to the decision boundary and points
toward the Non-Setosa class.

The margin represents the distance
between the hyperplane and the
nearest support vectors.
    """

    plt.text(0.5, 0.1, interpretation,
             fontsize=9, ha='center', va='top', style='italic',
             transform=ax.transAxes)


def _draw_weight_bars(w, ax, feature_names):
    """Helper function to draw weight vector component bars"""
    bar_width = 0.15
    bar_positions = [0.3, 0.6]
    colors_bars = ['steelblue', 'coral']
    labels = [f'w₁ = {w[0]:.4f}', f'w₂ = {w[1]:.4f}']

    for i, (pos, color, label) in enumerate(zip(bar_positions, colors_bars, labels)):
        rect_height = abs(w[i]) * 0.3  # Scale for visualization
        rect = plt.Rectangle((pos, 0.55 - rect_height/2), bar_width, rect_height,
                            facecolor=color, edgecolor='black', linewidth=2,
                            transform=ax.transAxes)
        ax.add_patch(rect)
        plt.text(pos + bar_width/2, 0.50, label,
                ha='center', va='top', fontsize=10, fontweight='bold',
                transform=ax.transAxes)
