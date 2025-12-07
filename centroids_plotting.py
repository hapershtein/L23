#!/usr/bin/env python3
"""
Plotting functions for centroid visualization.
Creates comprehensive visualizations of class centroids and distances.
"""

import numpy as np
import matplotlib.pyplot as plt
from centroids_data import create_distance_matrix


def plot_comprehensive_centroids(classes_data, class_names, centroids, feature_names,
                                 filename='centroids_visualization.png'):
    """Create comprehensive 3-panel centroid visualization"""
    fig = plt.figure(figsize=(16, 6))
    colors = ['red', 'blue', 'green']

    # Plot 1: All data points with centroids
    plt.subplot(1, 3, 1)
    _plot_data_with_centroids(classes_data, class_names, centroids, colors, feature_names)

    # Plot 2: Centroids only with clear labels
    plt.subplot(1, 3, 2)
    _plot_centroids_only(class_names, centroids, colors, feature_names)

    # Plot 3: Distance matrix visualization
    plt.subplot(1, 3, 3)
    _plot_distance_matrix(class_names, centroids, colors)

    plt.suptitle('Iris Dataset: Class Centroids on XY Plane\n(Petal Length vs Petal Width)',
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as '{filename}'")
    plt.close()


def _plot_data_with_centroids(classes_data, class_names, centroids, colors, feature_names):
    """Plot all data points with centroids"""
    for idx, (class_data, class_name, color) in enumerate(zip(classes_data, class_names, colors)):
        # Plot all data points
        plt.scatter(class_data[:, 0], class_data[:, 1],
                   c=color, label=class_name, alpha=0.4,
                   edgecolors='k', s=50, linewidth=0.5)

        # Plot centroid
        plt.scatter(centroids[idx][0], centroids[idx][1],
                   c=color, marker='*', s=800, edgecolors='black',
                   linewidth=3, label=f'{class_name} Centroid', zorder=10)

    # Draw lines connecting centroids
    for i in range(3):
        for j in range(i + 1, 3):
            plt.plot([centroids[i][0], centroids[j][0]],
                    [centroids[i][1], centroids[j][1]],
                    'k--', linewidth=2, alpha=0.5, zorder=5)

            # Add distance labels
            mid_x = (centroids[i][0] + centroids[j][0]) / 2
            mid_y = (centroids[i][1] + centroids[j][1]) / 2
            dist = np.linalg.norm(centroids[i] - centroids[j])
            plt.text(mid_x, mid_y, f'{dist:.3f}',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=1.5))

    plt.xlabel(feature_names[0], fontsize=12, fontweight='bold')
    plt.ylabel(feature_names[1], fontsize=12, fontweight='bold')
    plt.title('Data Points with Class Centroids', fontsize=13, fontweight='bold')
    plt.legend(loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)


def _plot_centroids_only(class_names, centroids, colors, feature_names):
    """Plot centroids only with labels and distances"""
    for idx, (class_name, color, centroid) in enumerate(zip(class_names, colors, centroids)):
        # Plot centroid
        plt.scatter(centroid[0], centroid[1],
                   c=color, marker='*', s=1000, edgecolors='black',
                   linewidth=3, label=class_name, zorder=10)

        # Add text label with coordinates
        offset_y = 0.15 if idx != 1 else -0.25
        plt.text(centroid[0], centroid[1] + offset_y,
                f'{class_name}\n({centroid[0]:.2f}, {centroid[1]:.2f})',
                fontsize=10, fontweight='bold', ha='center',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3, edgecolor='black'))

    # Draw lines connecting centroids with distance labels
    for i in range(3):
        for j in range(i + 1, 3):
            plt.plot([centroids[i][0], centroids[j][0]],
                    [centroids[i][1], centroids[j][1]],
                    'k-', linewidth=2, alpha=0.7, zorder=5)

            # Add distance labels
            mid_x = (centroids[i][0] + centroids[j][0]) / 2
            mid_y = (centroids[i][1] + centroids[j][1]) / 2
            dist = np.linalg.norm(centroids[i] - centroids[j])

            # Color code the distance label
            label_color = 'green' if dist < 2.0 else 'red'
            plt.text(mid_x, mid_y, f'd = {dist:.3f}',
                    fontsize=11, fontweight='bold', color=label_color,
                    bbox=dict(boxstyle='round', facecolor='yellow',
                             edgecolor=label_color, linewidth=2.5, alpha=0.8))

    plt.xlabel(feature_names[0], fontsize=12, fontweight='bold')
    plt.ylabel(feature_names[1], fontsize=12, fontweight='bold')
    plt.title('Class Centroids with Distances', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Adjust axis limits for better visibility
    x_min = min(c[0] for c in centroids) - 1
    x_max = max(c[0] for c in centroids) + 1
    y_min = min(c[1] for c in centroids) - 0.5
    y_max = max(c[1] for c in centroids) + 0.5
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)


def _plot_distance_matrix(class_names, centroids, colors):
    """Plot distance matrix as a table"""
    ax = plt.gca()
    ax.axis('off')

    dist_matrix = create_distance_matrix(centroids)

    # Create table
    title_text = "Distance Matrix Between Centroids\n"
    plt.text(0.5, 0.95, title_text, fontsize=13, fontweight='bold',
            ha='center', transform=ax.transAxes)

    # Header row
    y_pos = 0.80
    x_positions = [0.15, 0.40, 0.65, 0.90]

    # Column headers
    plt.text(x_positions[0], y_pos, "From \\ To", fontsize=10, fontweight='bold',
            ha='center', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightgray', edgecolor='black'))

    for i, class_name in enumerate(class_names):
        plt.text(x_positions[i+1], y_pos, class_name[:3], fontsize=10, fontweight='bold',
                ha='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.3, edgecolor='black'))

    # Data rows
    y_pos = 0.65
    for i, class_name in enumerate(class_names):
        # Row header
        plt.text(x_positions[0], y_pos, class_name[:3], fontsize=10, fontweight='bold',
                ha='center', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor=colors[i], alpha=0.3, edgecolor='black'))

        # Distance values
        for j in range(3):
            if i == j:
                text = "0.000"
                bgcolor = 'lightgray'
            else:
                text = f"{dist_matrix[i][j]:.3f}"
                # Highlight smallest distance
                bgcolor = 'lightgreen' if dist_matrix[i][j] < 2.0 else 'lightyellow'

            plt.text(x_positions[j+1], y_pos, text, fontsize=10,
                    ha='center', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor=bgcolor, edgecolor='black'))

        y_pos -= 0.15

    # Add legend
    legend_y = 0.30
    plt.text(0.5, legend_y, "Interpretation:", fontsize=11, fontweight='bold',
            ha='center', transform=ax.transAxes)

    legend_text = """
Smallest Distance (Green):
  Versicolor ↔ Virginica
  Most Similar Classes

Largest Distance (Red):
  Setosa ↔ Virginica
  Most Different Classes
    """

    plt.text(0.5, legend_y - 0.15, legend_text, fontsize=9,
            ha='center', va='top', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow',
                     edgecolor='black', linewidth=2))


def plot_simple_centroids(classes_data, class_names, centroids, feature_names,
                         filename='centroids_simple.png'):
    """Create a simplified single plot version"""
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['red', 'blue', 'green']

    # Plot all data points with transparency
    for idx, (class_data, class_name, color) in enumerate(zip(classes_data, class_names, colors)):
        ax.scatter(class_data[:, 0], class_data[:, 1],
                  c=color, label=f'{class_name} (data)', alpha=0.3,
                  edgecolors='k', s=60, linewidth=0.5)

    # Plot centroids as large stars
    for idx, (class_name, color, centroid) in enumerate(zip(class_names, colors, centroids)):
        ax.scatter(centroid[0], centroid[1],
                  c=color, marker='*', s=1500, edgecolors='black',
                  linewidth=4, label=f'{class_name} Centroid', zorder=10)

        # Add text annotation
        offset_y = 0.2 if idx != 1 else -0.3
        ax.annotate(f'{class_name}\nCentroid\n({centroid[0]:.2f}, {centroid[1]:.2f})',
                   xy=(centroid[0], centroid[1]),
                   xytext=(centroid[0], centroid[1] + offset_y),
                   fontsize=11, fontweight='bold', ha='center',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.4,
                            edgecolor='black', linewidth=2),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Draw lines connecting centroids
    for i in range(3):
        for j in range(i + 1, 3):
            ax.plot([centroids[i][0], centroids[j][0]],
                   [centroids[i][1], centroids[j][1]],
                   'k--', linewidth=2.5, alpha=0.6, zorder=5)

            # Add distance labels
            mid_x = (centroids[i][0] + centroids[j][0]) / 2
            mid_y = (centroids[i][1] + centroids[j][1]) / 2
            dist = np.linalg.norm(centroids[i] - centroids[j])

            # Color code based on distance
            if dist < 2.0:
                label_color = 'green'
                label_text = f'SMALLEST\nd = {dist:.3f}'
            else:
                label_color = 'darkred'
                label_text = f'd = {dist:.3f}'

            ax.text(mid_x, mid_y, label_text,
                   fontsize=10, fontweight='bold', color=label_color,
                   ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='yellow',
                            edgecolor=label_color, linewidth=2.5, alpha=0.9))

    ax.set_xlabel(feature_names[0], fontsize=13, fontweight='bold')
    ax.set_ylabel(feature_names[1], fontsize=13, fontweight='bold')
    ax.set_title('Iris Dataset: Class Centroids on XY Plane\nPetal Length vs Petal Width',
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

    # Add text box with key insight
    textstr = ('Key Insight:\n'
              'Versicolor & Virginica centroids\n'
              'are CLOSEST (d=1.636)\n'
              '→ Best candidates for grouping')
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, edgecolor='green', linewidth=3)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props, fontweight='bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Simplified visualization saved as '{filename}'")
    plt.close()
