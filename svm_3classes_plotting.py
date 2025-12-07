#!/usr/bin/env python3
"""
Visualization functions for 3-class SVM classification.
Creates plots for decision boundaries, support vectors, and confusion matrices.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(X, y, model, scaler, feature_names, class_names,
                          filename='svm_visualization_3classes.png'):
    """Plot decision boundary, support vectors, and data points for 3-class classification"""
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
    plt.figure(figsize=(16, 6))

    # Define colors for 3 classes
    colors = ['red', 'blue', 'green']
    cmap = plt.cm.get_cmap('RdYlGn', 3)

    # Plot 1: Decision boundary with support vectors (scaled)
    plt.subplot(1, 3, 1)
    _plot_scaled_boundaries(xx, yy, Z, X_scaled, y, model, class_names,
                           feature_names, colors, cmap)

    # Plot 2: Original data with support vectors marked
    plt.subplot(1, 3, 2)
    _plot_original_data(X, y, model, class_names, feature_names, colors)

    # Plot 3: Support vectors distribution by class
    plt.subplot(1, 3, 3)
    _plot_support_vectors_distribution(model, class_names, colors)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as '{filename}'")
    plt.close()


def _plot_scaled_boundaries(xx, yy, Z, X_scaled, y, model, class_names,
                           feature_names, colors, cmap):
    """Helper function to plot scaled decision boundaries"""
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap, levels=[-.5, .5, 1.5, 2.5])
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5, linestyles='dashed', levels=[0, 1, 2])

    # Plot the training points
    for idx, class_name in enumerate(class_names):
        plt.scatter(X_scaled[y == idx, 0], X_scaled[y == idx, 1],
                   c=colors[idx], label=class_name,
                   edgecolors='k', s=80, alpha=0.7)

    # Plot support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=200, linewidth=2, facecolors='none', edgecolors='black',
                label='Support Vectors')

    plt.xlabel(f'{feature_names[0]} (scaled)', fontsize=11)
    plt.ylabel(f'{feature_names[1]} (scaled)', fontsize=11)
    plt.title('SVM Decision Boundaries (Scaled)', fontsize=13, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)


def _plot_original_data(X, y, model, class_names, feature_names, colors):
    """Helper function to plot original data with support vectors"""
    for idx, class_name in enumerate(class_names):
        plt.scatter(X[y == idx, 0], X[y == idx, 1],
                   c=colors[idx], label=class_name,
                   edgecolors='k', s=80, alpha=0.7)

    # Mark support vectors
    support_vector_indices = model.support_
    plt.scatter(X[support_vector_indices, 0], X[support_vector_indices, 1],
                s=200, linewidth=2, facecolors='none', edgecolors='black',
                label=f'Support Vectors (n={len(support_vector_indices)})')

    plt.xlabel(feature_names[0], fontsize=11)
    plt.ylabel(feature_names[1], fontsize=11)
    plt.title('Original Data with Support Vectors', fontsize=13, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)


def _plot_support_vectors_distribution(model, class_names, colors):
    """Helper function to plot support vectors distribution"""
    n_support = model.n_support_
    bars = plt.bar(class_names, n_support, color=colors, edgecolor='black', linewidth=1.5)
    plt.xlabel('Class', fontsize=11)
    plt.ylabel('Number of Support Vectors', fontsize=11)
    plt.title('Support Vectors per Class', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')


def plot_confusion_matrix_heatmap(cm, class_names, filename='confusion_matrix_3classes.png'):
    """Create a heatmap visualization of the confusion matrix"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix Heatmap', fontsize=14, fontweight='bold')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix heatmap saved as '{filename}'")
    plt.close()
