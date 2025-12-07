#!/usr/bin/env python3
"""
Visualization functions for class separation analysis.
Creates comprehensive plots showing class distributions and separability.
"""

import numpy as np
import matplotlib.pyplot as plt


def create_separation_visualization(X, y, feature_names, class_names,
                                   filename='class_separation_analysis.png'):
    """Create comprehensive visualization showing class separation"""
    fig = plt.figure(figsize=(16, 10))

    colors = ['red', 'blue', 'green']

    # Plot 1-4: Individual feature distributions
    _plot_feature_distributions(X, y, feature_names, class_names, colors)

    # Plot 5: Petal Length vs Petal Width (best separation)
    plt.subplot(3, 3, 5)
    _plot_petal_scatter(X, y, class_names, colors)

    # Plot 6: Sepal Length vs Sepal Width (more overlap)
    plt.subplot(3, 3, 6)
    _plot_sepal_scatter(X, y, class_names, colors)

    # Plot 7: Class centroid distances
    plt.subplot(3, 3, 7)
    _plot_centroid_distances(X, y, class_names, colors)

    # Plot 8: Mean feature values by class
    plt.subplot(3, 3, 8)
    _plot_mean_features(X, y, feature_names, class_names, colors)

    # Plot 9: Summary text
    plt.subplot(3, 3, 9)
    _plot_summary_text()

    plt.suptitle('Iris Dataset: Class Separation Analysis', fontsize=14,
                fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved as '{filename}'")
    plt.close()


def _plot_feature_distributions(X, y, feature_names, class_names, colors):
    """Plot individual feature distributions"""
    for feat_idx in range(4):
        plt.subplot(3, 3, feat_idx + 1)

        for class_idx in range(3):
            class_data = X[y == class_idx, feat_idx]
            plt.hist(class_data, bins=15, alpha=0.6, color=colors[class_idx],
                    label=class_names[class_idx], edgecolor='black')

        plt.xlabel(feature_names[feat_idx], fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.title(f'Distribution: {feature_names[feat_idx]}', fontsize=11, fontweight='bold')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)


def _plot_petal_scatter(X, y, class_names, colors):
    """Plot Petal Length vs Petal Width scatter"""
    for class_idx, class_name in enumerate(class_names):
        mask = y == class_idx
        plt.scatter(X[mask, 2], X[mask, 3], c=colors[class_idx],
                   label=class_name, alpha=0.7, edgecolors='k', s=60)

    plt.xlabel('Petal Length (cm)', fontsize=10)
    plt.ylabel('Petal Width (cm)', fontsize=10)
    plt.title('Petal Length vs Petal Width\n(Best Separation)', fontsize=11, fontweight='bold')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # Add annotation showing Versicolor-Virginica overlap
    plt.annotate('Versicolor-Virginica\nOverlap Region',
                xy=(5.0, 1.5), fontsize=9, color='purple',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))


def _plot_sepal_scatter(X, y, class_names, colors):
    """Plot Sepal Length vs Sepal Width scatter"""
    for class_idx, class_name in enumerate(class_names):
        mask = y == class_idx
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[class_idx],
                   label=class_name, alpha=0.7, edgecolors='k', s=60)

    plt.xlabel('Sepal Length (cm)', fontsize=10)
    plt.ylabel('Sepal Width (cm)', fontsize=10)
    plt.title('Sepal Length vs Sepal Width\n(More Overlap)', fontsize=11, fontweight='bold')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)


def _plot_centroid_distances(X, y, class_names, colors):
    """Plot class centroid distances"""
    centroids = []
    for class_idx in range(3):
        centroid = np.mean(X[y == class_idx], axis=0)
        centroids.append(centroid)

    distance_pairs = []
    distance_values = []
    pair_colors = []

    for i in range(3):
        for j in range(i + 1, 3):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            pair_label = f"{class_names[i][:3]}-{class_names[j][:3]}"
            distance_pairs.append(pair_label)
            distance_values.append(dist)

            # Color code: smallest distance in green
            if dist == min([np.linalg.norm(centroids[a] - centroids[b])
                           for a in range(3) for b in range(a+1, 3)]):
                pair_colors.append('lightgreen')
            else:
                pair_colors.append('lightcoral')

    bars = plt.bar(distance_pairs, distance_values, color=pair_colors,
                   edgecolor='black', linewidth=2)
    plt.ylabel('Euclidean Distance', fontsize=10)
    plt.title('Centroid Distances\n(Smaller = More Similar)', fontsize=11, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, distance_values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Highlight the smallest
    plt.annotate('SMALLEST\n(Most Similar)',
                xy=(0, distance_values[0]), xytext=(0.5, distance_values[0] + 2),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=9, color='green', fontweight='bold')


def _plot_mean_features(X, y, feature_names, class_names, colors):
    """Plot mean feature values by class"""
    # Calculate mean feature values for each class
    feature_means = np.array([[np.mean(X[y == class_idx, feat_idx])
                              for feat_idx in range(4)]
                             for class_idx in range(3)])

    x = np.arange(4)
    width = 0.25

    for class_idx in range(3):
        offset = (class_idx - 1) * width
        plt.bar(x + offset, feature_means[class_idx], width,
               label=class_names[class_idx], color=colors[class_idx],
               alpha=0.7, edgecolor='black')

    plt.xlabel('Features', fontsize=10)
    plt.ylabel('Mean Value (cm)', fontsize=10)
    plt.title('Mean Feature Values by Class', fontsize=11, fontweight='bold')
    plt.xticks(x, ['Sepal L', 'Sepal W', 'Petal L', 'Petal W'], fontsize=8)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3, axis='y')


def _plot_summary_text():
    """Plot summary text box"""
    plt.axis('off')

    summary_text = """
GROUPING DECISION

Why Versicolor + Virginica?

✓ Smallest centroid distance
✓ Significant feature overlap
✓ Similar petal dimensions
✓ Biologically related species

Why NOT other groupings?

✗ Setosa is clearly separated
✗ Setosa + others = unnatural
✗ Would reduce interpretability

CONCLUSION:
Group Versicolor and Virginica
as "Non-Setosa" for optimal
binary classification.
    """

    plt.text(0.5, 0.5, summary_text, fontsize=10,
            ha='center', va='center', transform=plt.gca().transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow',
                     edgecolor='black', linewidth=2),
            family='monospace')
