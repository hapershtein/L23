#!/usr/bin/env python3
"""
Analysis of Iris Class Separation
This script analyzes why Versicolor and Virginica were grouped together
by examining the statistical properties and separability of the three classes.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial.distance import cdist

def analyze_class_separation():
    """Analyze and visualize why Versicolor and Virginica are similar"""
    # Load iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names

    print("=" * 80)
    print("Analysis: Why Group Versicolor and Virginica Together?")
    print("=" * 80)
    print()

    # Separate data by class
    setosa = X[y == 0]
    versicolor = X[y == 1]
    virginica = X[y == 2]

    classes_data = [setosa, versicolor, virginica]

    # 1. Statistical Analysis
    print("1. STATISTICAL PROPERTIES PER CLASS")
    print("-" * 80)
    print()

    for idx, (class_data, class_name) in enumerate(zip(classes_data, class_names)):
        print(f"{class_name.upper()}:")
        for feat_idx, feat_name in enumerate(feature_names):
            mean = np.mean(class_data[:, feat_idx])
            std = np.std(class_data[:, feat_idx])
            print(f"  {feat_name:20s}: mean={mean:.2f}, std={std:.2f}")
        print()

    # 2. Pairwise Distance Analysis
    print("\n2. INTER-CLASS DISTANCE ANALYSIS")
    print("-" * 80)
    print()
    print("Average distance between class centroids (using all 4 features):\n")

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
            print(f"  {pair:30s}: {dist:.4f}")

    print()
    print("Interpretation:")
    min_pair = min(distances, key=distances.get)
    max_pair = max(distances, key=distances.get)
    print(f"  - CLOSEST classes: {min_pair} (distance: {distances[min_pair]:.4f})")
    print(f"  - FARTHEST classes: {max_pair} (distance: {distances[max_pair]:.4f})")
    print()

    # 3. Feature-wise Overlap Analysis
    print("\n3. FEATURE-WISE OVERLAP ANALYSIS")
    print("-" * 80)
    print()
    print("Measuring overlap between classes for each feature:\n")

    for feat_idx, feat_name in enumerate(feature_names):
        print(f"{feat_name}:")
        setosa_vals = setosa[:, feat_idx]
        versicolor_vals = versicolor[:, feat_idx]
        virginica_vals = virginica[:, feat_idx]

        # Calculate ranges
        setosa_range = (setosa_vals.min(), setosa_vals.max())
        versicolor_range = (versicolor_vals.min(), versicolor_vals.max())
        virginica_range = (virginica_vals.min(), virginica_vals.max())

        print(f"  Setosa range:     [{setosa_range[0]:.2f}, {setosa_range[1]:.2f}]")
        print(f"  Versicolor range: [{versicolor_range[0]:.2f}, {versicolor_range[1]:.2f}]")
        print(f"  Virginica range:  [{virginica_range[0]:.2f}, {virginica_range[1]:.2f}]")

        # Check overlaps
        overlap_sv = max(0, min(setosa_range[1], versicolor_range[1]) - max(setosa_range[0], versicolor_range[0]))
        overlap_si = max(0, min(setosa_range[1], virginica_range[1]) - max(setosa_range[0], virginica_range[0]))
        overlap_vi = max(0, min(versicolor_range[1], virginica_range[1]) - max(versicolor_range[0], virginica_range[0]))

        print(f"  Overlap Setosa-Versicolor: {overlap_sv:.2f}")
        print(f"  Overlap Setosa-Virginica:  {overlap_si:.2f}")
        print(f"  Overlap Versicolor-Virginica: {overlap_vi:.2f}")
        print()

    # 4. Create comprehensive visualization
    create_separation_visualization(X, y, feature_names, class_names)

    # 5. Summary and Recommendation
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

def create_separation_visualization(X, y, feature_names, class_names):
    """Create comprehensive visualization showing class separation"""
    fig = plt.figure(figsize=(16, 10))

    colors = ['red', 'blue', 'green']

    # Plot 1-4: Individual feature distributions
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

    # Plot 5: Petal Length vs Petal Width (best separation)
    plt.subplot(3, 3, 5)
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

    # Plot 6: Sepal Length vs Sepal Width (more overlap)
    plt.subplot(3, 3, 6)
    for class_idx, class_name in enumerate(class_names):
        mask = y == class_idx
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[class_idx],
                   label=class_name, alpha=0.7, edgecolors='k', s=60)

    plt.xlabel('Sepal Length (cm)', fontsize=10)
    plt.ylabel('Sepal Width (cm)', fontsize=10)
    plt.title('Sepal Length vs Sepal Width\n(More Overlap)', fontsize=11, fontweight='bold')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # Plot 7: Class centroid distances
    plt.subplot(3, 3, 7)

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

    # Plot 8: Box plots showing feature distributions
    plt.subplot(3, 3, 8)

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

    # Plot 9: Summary text
    plt.subplot(3, 3, 9)
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

    plt.suptitle('Iris Dataset: Class Separation Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('class_separation_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'class_separation_analysis.png'")
    plt.close()

if __name__ == "__main__":
    analyze_class_separation()
