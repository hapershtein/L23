#!/usr/bin/env python3
"""
Visualization functions for binary SVM classification.
Creates plots for decision boundaries and support vectors.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(X, y, model, scaler, feature_names, class_names,
                          filename='svm_visualization.png'):
    """Plot decision boundary, support vectors, and data points"""
    # Scale the data
    X_scaled = scaler.transform(X)

    # Create a mesh to plot decision boundary
    h = 0.02  # step size in the mesh
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create the plot
    plt.figure(figsize=(12, 5))

    # Plot 1: Decision boundary with support vectors
    plt.subplot(1, 2, 1)
    _plot_scaled_decision_boundary(xx, yy, Z, X_scaled, y, model, feature_names)

    # Plot 2: Original data with support vectors marked
    plt.subplot(1, 2, 2)
    _plot_original_data(X, y, model, class_names, feature_names)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as '{filename}'")
    plt.close()


def _plot_scaled_decision_boundary(xx, yy, Z, X_scaled, y, model, feature_names):
    """Plot decision boundary in scaled feature space"""
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5, linestyles='dashed')

    # Plot the training points
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y,
                         cmap=plt.cm.RdYlBu, edgecolors='k', s=100)

    # Plot support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=200, linewidth=2, facecolors='none', edgecolors='green',
                label='Support Vectors')

    plt.xlabel(f'{feature_names[0]} (scaled)', fontsize=12)
    plt.ylabel(f'{feature_names[1]} (scaled)', fontsize=12)
    plt.title('SVM Decision Boundary with Support Vectors', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)


def _plot_original_data(X, y, model, class_names, feature_names):
    """Plot original data with support vectors marked"""
    colors = ['red', 'blue']
    for idx, class_name in enumerate(class_names):
        plt.scatter(X[y == idx, 0], X[y == idx, 1],
                   c=colors[idx], label=class_name,
                   edgecolors='k', s=100, alpha=0.7)

    # Mark support vectors
    support_vector_indices = model.support_
    plt.scatter(X[support_vector_indices, 0], X[support_vector_indices, 1],
                s=200, linewidth=2, facecolors='none', edgecolors='green',
                label=f'Support Vectors (n={len(support_vector_indices)})')

    plt.xlabel(feature_names[0], fontsize=12)
    plt.ylabel(feature_names[1], fontsize=12)
    plt.title('Original Data with Support Vectors', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
