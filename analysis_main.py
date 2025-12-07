#!/usr/bin/env python3
"""
Main execution script for class separation analysis.
Analyzes why Versicolor and Virginica were grouped together.
"""

from analysis_data import (
    load_iris_data,
    separate_by_class,
    compute_class_statistics,
    compute_centroid_distances,
    compute_feature_overlaps,
    print_statistical_analysis,
    print_distance_analysis,
    print_overlap_analysis,
    print_grouping_recommendation
)
from analysis_visualization import create_separation_visualization


def analyze_class_separation():
    """Analyze and visualize why Versicolor and Virginica are similar"""
    # Load iris dataset
    X, y, feature_names, class_names = load_iris_data()

    print("=" * 80)
    print("Analysis: Why Group Versicolor and Virginica Together?")
    print("=" * 80)
    print()

    # Separate data by class
    classes_data = separate_by_class(X, y)

    # 1. Statistical Analysis
    stats = compute_class_statistics(classes_data, feature_names, class_names)
    print_statistical_analysis(stats)

    # 2. Pairwise Distance Analysis
    distances, centroids = compute_centroid_distances(classes_data, class_names)
    print_distance_analysis(distances)

    # 3. Feature-wise Overlap Analysis
    overlaps = compute_feature_overlaps(X, y, feature_names)
    print_overlap_analysis(overlaps)

    # 4. Create comprehensive visualization
    create_separation_visualization(X, y, feature_names, class_names)

    # 5. Summary and Recommendation
    print_grouping_recommendation()


if __name__ == "__main__":
    analyze_class_separation()
