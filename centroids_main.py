#!/usr/bin/env python3
"""
Main execution script for centroid visualization.
Plots centroids of the three Iris classes on a 2D plane.
"""

from centroids_data import (
    load_iris_for_centroids,
    separate_classes,
    calculate_centroids,
    calculate_pairwise_distances,
    print_centroid_info
)
from centroids_plotting import (
    plot_comprehensive_centroids,
    plot_simple_centroids
)


def plot_centroids_2d():
    """Plot centroids of three classes on XY plane"""
    # Load iris dataset
    X, y, feature_names, class_names = load_iris_for_centroids()

    # Separate data by class
    classes_data = separate_classes(X, y)

    # Calculate centroids
    centroids = calculate_centroids(classes_data)

    # Calculate pairwise distances
    distances = calculate_pairwise_distances(centroids)

    # Print information
    print_centroid_info(class_names, centroids, distances)

    # Create the main comprehensive visualization
    plot_comprehensive_centroids(classes_data, class_names, centroids, feature_names)

    # Create a simplified single plot version
    plot_simple_centroids(classes_data, class_names, centroids, feature_names)

    print()
    print("=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    plot_centroids_2d()
