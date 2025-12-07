#!/usr/bin/env python3
"""
Data loading and statistical analysis functions for class separation analysis.
"""

import numpy as np
from sklearn import datasets
from scipy.spatial.distance import cdist


def load_iris_data():
    """Load Iris dataset and return data, labels, feature names, and class names"""
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names

    return X, y, feature_names, class_names


def separate_by_class(X, y):
    """Separate data by class"""
    setosa = X[y == 0]
    versicolor = X[y == 1]
    virginica = X[y == 2]

    return [setosa, versicolor, virginica]


def compute_class_statistics(classes_data, feature_names, class_names):
    """Compute mean and standard deviation for each feature per class"""
    stats = {}

    for idx, (class_data, class_name) in enumerate(zip(classes_data, class_names)):
        class_stats = {}
        for feat_idx, feat_name in enumerate(feature_names):
            mean = np.mean(class_data[:, feat_idx])
            std = np.std(class_data[:, feat_idx])
            class_stats[feat_name] = {'mean': mean, 'std': std}
        stats[class_name] = class_stats

    return stats


def compute_centroid_distances(classes_data, class_names):
    """Compute pairwise distances between class centroids"""
    centroids = []
    for class_data in classes_data:
        centroid = np.mean(class_data, axis=0)
        centroids.append(centroid)

    distances = {}
    for i in range(3):
        for j in range(i + 1, 3):
            dist = np.linalg.norm(centroids[i] - centroids[j])
            pair = f"{class_names[i]} <-> {class_names[j]}"
            distances[pair] = dist

    return distances, centroids


def compute_feature_overlaps(X, y, feature_names):
    """Compute feature-wise overlap between classes"""
    setosa = X[y == 0]
    versicolor = X[y == 1]
    virginica = X[y == 2]

    overlaps = {}

    for feat_idx, feat_name in enumerate(feature_names):
        setosa_vals = setosa[:, feat_idx]
        versicolor_vals = versicolor[:, feat_idx]
        virginica_vals = virginica[:, feat_idx]

        # Calculate ranges
        setosa_range = (setosa_vals.min(), setosa_vals.max())
        versicolor_range = (versicolor_vals.min(), versicolor_vals.max())
        virginica_range = (virginica_vals.min(), virginica_vals.max())

        # Check overlaps
        overlap_sv = max(0, min(setosa_range[1], versicolor_range[1]) -
                        max(setosa_range[0], versicolor_range[0]))
        overlap_si = max(0, min(setosa_range[1], virginica_range[1]) -
                        max(setosa_range[0], virginica_range[0]))
        overlap_vi = max(0, min(versicolor_range[1], virginica_range[1]) -
                        max(versicolor_range[0], virginica_range[0]))

        overlaps[feat_name] = {
            'setosa_range': setosa_range,
            'versicolor_range': versicolor_range,
            'virginica_range': virginica_range,
            'overlap_setosa_versicolor': overlap_sv,
            'overlap_setosa_virginica': overlap_si,
            'overlap_versicolor_virginica': overlap_vi
        }

    return overlaps


def print_statistical_analysis(stats):
    """Print statistical properties per class"""
    print("1. STATISTICAL PROPERTIES PER CLASS")
    print("-" * 80)
    print()

    for class_name, class_stats in stats.items():
        print(f"{class_name.upper()}:")
        for feat_name, feat_stats in class_stats.items():
            mean = feat_stats['mean']
            std = feat_stats['std']
            print(f"  {feat_name:20s}: mean={mean:.2f}, std={std:.2f}")
        print()


def print_distance_analysis(distances):
    """Print inter-class distance analysis"""
    print("\n2. INTER-CLASS DISTANCE ANALYSIS")
    print("-" * 80)
    print()
    print("Average distance between class centroids (using all 4 features):\n")

    for pair, dist in distances.items():
        print(f"  {pair:30s}: {dist:.4f}")

    print()
    print("Interpretation:")
    min_pair = min(distances, key=distances.get)
    max_pair = max(distances, key=distances.get)
    print(f"  - CLOSEST classes: {min_pair} (distance: {distances[min_pair]:.4f})")
    print(f"  - FARTHEST classes: {max_pair} (distance: {distances[max_pair]:.4f})")
    print()


def print_overlap_analysis(overlaps):
    """Print feature-wise overlap analysis"""
    print("\n3. FEATURE-WISE OVERLAP ANALYSIS")
    print("-" * 80)
    print()
    print("Measuring overlap between classes for each feature:\n")

    for feat_name, overlap_data in overlaps.items():
        print(f"{feat_name}:")
        print(f"  Setosa range:     [{overlap_data['setosa_range'][0]:.2f}, {overlap_data['setosa_range'][1]:.2f}]")
        print(f"  Versicolor range: [{overlap_data['versicolor_range'][0]:.2f}, {overlap_data['versicolor_range'][1]:.2f}]")
        print(f"  Virginica range:  [{overlap_data['virginica_range'][0]:.2f}, {overlap_data['virginica_range'][1]:.2f}]")
        print(f"  Overlap Setosa-Versicolor: {overlap_data['overlap_setosa_versicolor']:.2f}")
        print(f"  Overlap Setosa-Virginica:  {overlap_data['overlap_setosa_virginica']:.2f}")
        print(f"  Overlap Versicolor-Virginica: {overlap_data['overlap_versicolor_virginica']:.2f}")
        print()


def print_grouping_recommendation():
    """Print grouping recommendation summary"""
    print("\n4. GROUPING RECOMMENDATION")
    print("-" * 80)
    print()
    print("DECISION: Group Versicolor and Virginica together")
    print()
    print("REASONS:")
    print("  1. Versicolor and Virginica have the SMALLEST centroid distance")
    print("  2. These two classes show significant feature overlap, especially in")
    print("     sepal measurements")
    print("  3. Setosa is clearly separated from both other classes across all features")
    print("  4. Biologically, Versicolor and Virginica are more closely related species")
    print()
    print("ALTERNATIVE GROUPINGS (less effective):")
    print("  - Grouping Setosa + Versicolor would create an unnatural boundary")
    print("  - Grouping Setosa + Virginica would also be less intuitive")
    print("  - Both alternatives ignore the natural similarity between Versicolor/Virginica")
    print()
    print("RESULT:")
    print("  By grouping Versicolor and Virginica, we create a clear binary")
    print("  classification: 'Setosa' vs 'Non-Setosa', which maximizes class")
    print("  separation and makes the weight vector analysis more meaningful.")
    print()
