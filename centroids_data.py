#!/usr/bin/env python3
"""
Data loading and centroid calculation functions for centroid visualization.
"""

import numpy as np
from sklearn import datasets


def load_iris_for_centroids():
    """Load Iris dataset with petal features"""
    iris = datasets.load_iris()
    X = iris.data[:, 2:4]  # Petal length and petal width
    y = iris.target
    feature_names = ['Petal Length (cm)', 'Petal Width (cm)']
    class_names = iris.target_names

    return X, y, feature_names, class_names


def separate_classes(X, y):
    """Separate data by class"""
    setosa = X[y == 0]
    versicolor = X[y == 1]
    virginica = X[y == 2]

    return [setosa, versicolor, virginica]


def calculate_centroids(classes_data):
    """Calculate centroids for each class"""
    centroids = []
    for class_data in classes_data:
        centroid = np.mean(class_data, axis=0)
        centroids.append(centroid)

    return centroids


def calculate_pairwise_distances(centroids):
    """Calculate pairwise distances between centroids"""
    distances = []
    for i in range(3):
        for j in range(i + 1, 3):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            distances.append((i, j, dist))

    return distances


def create_distance_matrix(centroids):
    """Create distance matrix between centroids"""
    dist_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i != j:
                dist_matrix[i][j] = np.linalg.norm(centroids[i] - centroids[j])

    return dist_matrix


def print_centroid_info(class_names, centroids, distances):
    """Print centroid coordinates and distances"""
    print("=" * 70)
    print("Iris Dataset - Centroid Analysis")
    print("=" * 70)
    print()
    print("Centroids (using Petal Length and Petal Width):")
    print()
    for i, (class_name, centroid) in enumerate(zip(class_names, centroids)):
        print(f"  {class_name:12s}: [{centroid[0]:.4f}, {centroid[1]:.4f}]")
    print()

    print("Distances between centroids:")
    print()
    for i, j, dist in distances:
        print(f"  {class_names[i]:12s} <-> {class_names[j]:12s}: {dist:.4f}")
    print()
