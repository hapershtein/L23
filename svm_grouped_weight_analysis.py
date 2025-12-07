#!/usr/bin/env python3
"""
Weight vector analysis and visualization for grouped SVM classification.
"""

import matplotlib.pyplot as plt
from svm_grouped_model import calculate_hyperplane_parameters


def plot_weight_vector_analysis(model, feature_names, filename='weight_vector_analysis.png'):
    """Create detailed analysis of weight vector"""
    w, b, margin, w_norm = calculate_hyperplane_parameters(model)

    plt.figure(figsize=(12, 5))

    # Plot 1: Weight vector components
    plt.subplot(1, 2, 1)
    _plot_weight_components(w, feature_names)

    # Plot 2: Hyperplane parameters summary
    plt.subplot(1, 2, 2)
    _plot_parameter_table(w, b, margin, w_norm, model, feature_names)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Weight vector analysis saved as '{filename}'")
    plt.close()


def _plot_weight_components(w, feature_names):
    """Plot weight vector components as bar chart"""
    components = [w[0], w[1]]
    labels = [f'{feature_names[0]}\n(w₁)', f'{feature_names[1]}\n(w₂)']
    colors = ['steelblue', 'coral']

    bars = plt.bar(labels, components, color=colors, edgecolor='black', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.ylabel('Weight Value', fontsize=12, fontweight='bold')
    plt.title('Weight Vector Components', fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, components):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=11, fontweight='bold')


def _plot_parameter_table(w, b, margin, w_norm, model, feature_names):
    """Plot hyperplane parameters as a formatted table"""
    ax = plt.gca()
    ax.axis('off')

    # Create parameter table
    param_data = [
        ['Parameter', 'Value', 'Meaning'],
        ['', '', ''],
        ['w₁', f'{w[0]:.6f}', f'Weight for {feature_names[0]}'],
        ['w₂', f'{w[1]:.6f}', f'Weight for {feature_names[1]}'],
        ['', '', ''],
        ['Bias (b)', f'{b:.6f}', 'Hyperplane offset'],
        ['||w||', f'{w_norm:.6f}', 'Magnitude of weight vector'],
        ['Margin', f'{margin:.6f}', 'Distance between support vectors'],
        ['', '', ''],
        ['# Support Vectors', f'{len(model.support_vectors_)}', 'Critical data points'],
    ]

    # Create table
    table = plt.table(cellText=param_data, cellLoc='left',
                     bbox=[0, 0, 1, 1], edges='horizontal')
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Style header row
    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, len(param_data)):
        for j in range(3):
            cell = table[(i, j)]
            if param_data[i][0] == '':  # Empty separator rows
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
            if j == 0:  # First column (parameter names)
                cell.set_text_props(weight='bold')

    plt.title('Hyperplane Parameters Summary', fontsize=13, fontweight='bold', pad=20)
