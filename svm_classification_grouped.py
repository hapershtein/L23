#!/usr/bin/env python3
"""
Binary Classification with Grouped Classes using Support Vector Machine on Iris Dataset
This script groups Versicolor and Virginica into one class and classifies against Setosa.
It calculates and visualizes the weight vector (w) and support vectors.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load Iris dataset and group Versicolor and Virginica into one class"""
    # Load iris dataset
    iris = datasets.load_iris()

    # Use only 2 features (petal length and petal width) for 2D visualization
    X = iris.data[:, 2:4]  # Petal length and petal width
    y = iris.target.copy()

    # Group classes: Setosa (0) vs Non-Setosa (1)
    # Original: Setosa=0, Versicolor=1, Virginica=2
    # New: Setosa=0, Non-Setosa (Versicolor + Virginica)=1
    y[y > 0] = 1

    feature_names = ['Petal Length (cm)', 'Petal Width (cm)']
    class_names = ['Setosa', 'Non-Setosa (Versicolor + Virginica)']
    original_classes = ['Setosa', 'Versicolor', 'Virginica']

    return X, y, feature_names, class_names, original_classes

def train_svm_model(X_train, y_train):
    """Train SVM model with linear kernel for binary classification"""
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train SVM with linear kernel to get interpretable weight vector
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    return svm_model, scaler

def calculate_hyperplane_parameters(model):
    """Calculate weight vector w, bias b, and margin"""
    # For linear SVM: decision_function = w·x + b
    w = model.coef_[0]  # Weight vector
    b = model.intercept_[0]  # Bias term

    # Calculate margin: 2/||w||
    w_norm = np.linalg.norm(w)
    margin = 2 / w_norm

    return w, b, margin, w_norm

def plot_decision_boundary_with_hyperplane(X, y, model, scaler, feature_names, class_names,
                                           original_classes, filename='svm_visualization_grouped.png'):
    """Plot decision boundary, hyperplane, support vectors, and weight vector"""
    # Scale the data
    X_scaled = scaler.transform(X)

    # Get hyperplane parameters
    w, b, margin, w_norm = calculate_hyperplane_parameters(model)

    # Create a mesh to plot decision boundary
    h = 0.02
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict for each point in the mesh
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Create the plot
    plt.figure(figsize=(16, 6))

    # Plot 1: Decision boundary with hyperplane and margins (scaled)
    plt.subplot(1, 3, 1)

    # Plot decision boundary (filled contours)
    plt.contourf(xx, yy, Z > 0, alpha=0.3, levels=1, cmap=plt.cm.RdYlBu)

    # Plot hyperplane (decision boundary) and margins
    plt.contour(xx, yy, Z, colors=['k', 'gray', 'gray'],
                levels=[-1, 0, 1], linestyles=['dashed', 'solid', 'dashed'],
                linewidths=[2, 3, 2])

    # Plot data points
    colors = ['red', 'blue']
    for idx, class_name in enumerate(class_names):
        mask = y == idx
        plt.scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                   c=colors[idx], label=class_name,
                   edgecolors='k', s=80, alpha=0.7)

    # Plot support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=200, linewidth=2, facecolors='none', edgecolors='green',
                label='Support Vectors')

    # Plot weight vector from origin
    scale_factor = 2.0
    origin_x = (x_min + x_max) / 2
    origin_y = (y_min + y_max) / 2
    plt.arrow(origin_x, origin_y, w[0] * scale_factor, w[1] * scale_factor,
              head_width=0.15, head_length=0.15, fc='purple', ec='purple',
              linewidth=3, label=f'Weight Vector w', zorder=5)

    plt.xlabel(f'{feature_names[0]} (scaled)', fontsize=11)
    plt.ylabel(f'{feature_names[1]} (scaled)', fontsize=11)
    plt.title('SVM Hyperplane and Weight Vector', fontsize=13, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)

    # Plot 2: Original data with original class labels
    plt.subplot(1, 3, 2)
    original_y = np.zeros(len(X))
    original_y[:50] = 0   # Setosa
    original_y[50:100] = 1  # Versicolor
    original_y[100:] = 2  # Virginica

    colors_original = ['red', 'blue', 'green']
    for idx, class_name in enumerate(original_classes):
        mask = original_y == idx
        plt.scatter(X[mask, 0], X[mask, 1],
                   c=colors_original[idx], label=class_name,
                   edgecolors='k', s=80, alpha=0.7)

    # Mark support vectors
    support_vector_indices = model.support_
    plt.scatter(X[support_vector_indices, 0], X[support_vector_indices, 1],
                s=200, linewidth=2, facecolors='none', edgecolors='black',
                label=f'Support Vectors (n={len(support_vector_indices)})')

    plt.xlabel(feature_names[0], fontsize=11)
    plt.ylabel(feature_names[1], fontsize=11)
    plt.title('Original 3 Classes with Support Vectors', fontsize=13, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)

    # Plot 3: Weight vector components and hyperplane parameters
    plt.subplot(1, 3, 3)

    # Create a table-like visualization
    ax = plt.gca()
    ax.axis('off')

    # Title
    plt.text(0.5, 0.95, 'Hyperplane Parameters',
             fontsize=14, fontweight='bold', ha='center', transform=ax.transAxes)

    # Weight vector visualization
    plt.text(0.5, 0.85, 'Weight Vector (w)',
             fontsize=12, fontweight='bold', ha='center', transform=ax.transAxes)

    # Draw weight vector components
    bar_width = 0.15
    bar_positions = [0.3, 0.6]
    colors_bars = ['steelblue', 'coral']
    labels = [f'w₁ = {w[0]:.4f}', f'w₂ = {w[1]:.4f}']

    for i, (pos, color, label) in enumerate(zip(bar_positions, colors_bars, labels)):
        rect_height = abs(w[i]) * 0.3  # Scale for visualization
        rect = plt.Rectangle((pos, 0.55 - rect_height/2), bar_width, rect_height,
                            facecolor=color, edgecolor='black', linewidth=2,
                            transform=ax.transAxes)
        ax.add_patch(rect)
        plt.text(pos + bar_width/2, 0.50, label,
                ha='center', va='top', fontsize=10, fontweight='bold',
                transform=ax.transAxes)

    # Parameters text
    params_text = f"""
Hyperplane Equation:
{w[0]:.4f} × x₁ + {w[1]:.4f} × x₂ + {b:.4f} = 0

||w|| = {w_norm:.4f}
Margin = 2/||w|| = {margin:.4f}

Support Vectors: {len(model.support_vectors_)}
    """

    plt.text(0.5, 0.35, params_text,
             fontsize=10, ha='center', va='top',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             transform=ax.transAxes)

    # Add interpretation
    interpretation = """
The weight vector w is perpendicular
to the decision boundary and points
toward the Non-Setosa class.

The margin represents the distance
between the hyperplane and the
nearest support vectors.
    """

    plt.text(0.5, 0.1, interpretation,
             fontsize=9, ha='center', va='top', style='italic',
             transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as '{filename}'")
    plt.close()

def plot_weight_vector_analysis(model, feature_names, filename='weight_vector_analysis.png'):
    """Create detailed analysis of weight vector"""
    w, b, margin, w_norm = calculate_hyperplane_parameters(model)

    plt.figure(figsize=(12, 5))

    # Plot 1: Weight vector components
    plt.subplot(1, 2, 1)
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

    # Plot 2: Hyperplane parameters summary
    plt.subplot(1, 2, 2)
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

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Weight vector analysis saved as '{filename}'")
    plt.close()

def generate_results_markdown(X_train, X_test, y_train, y_test, model, scaler,
                              feature_names, class_names, original_classes,
                              output_file='Results_grouped.md'):
    """Generate markdown file with results for grouped classification"""
    # Make predictions
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)

    # Classification report
    report = classification_report(y_test, y_test_pred, target_names=class_names)

    # Get SVM parameters
    n_support = model.n_support_
    support_vectors = model.support_vectors_

    # Get hyperplane parameters
    w, b, margin, w_norm = calculate_hyperplane_parameters(model)

    # Write to markdown file
    with open(output_file, 'w') as f:
        f.write("# Binary Classification with Grouped Classes using Support Vector Machine\n\n")

        f.write("## Overview\n\n")
        f.write("This analysis groups two of the three Iris classes together to perform binary classification:\n")
        f.write(f"- **Class 0**: {original_classes[0]}\n")
        f.write(f"- **Class 1**: {original_classes[1]} + {original_classes[2]} (grouped as 'Non-Setosa')\n\n")
        f.write("This grouping allows us to calculate and analyze the **weight vector (w)** and support vectors ")
        f.write("for a linear decision boundary.\n\n")

        f.write("## Dataset Information\n\n")
        f.write("- **Dataset**: Iris Dataset (Binary Classification with Grouped Classes)\n")
        f.write(f"- **Original Classes**: {', '.join(original_classes)}\n")
        f.write(f"- **Grouped Classes**: {class_names[0]} vs {class_names[1]}\n")
        f.write(f"- **Features Used**: {feature_names[0]}, {feature_names[1]}\n")
        f.write(f"- **Total Samples**: {len(X_train) + len(X_test)}\n")
        f.write(f"- **Training Samples**: {len(X_train)}\n")
        f.write(f"- **Testing Samples**: {len(X_test)}\n\n")

        f.write("## SVM Model Configuration\n\n")
        f.write("- **Kernel**: Linear (enables calculation of explicit weight vector)\n")
        f.write("- **C Parameter**: 1.0\n")
        f.write("- **Feature Scaling**: StandardScaler\n\n")

        f.write("## Hyperplane Parameters\n\n")
        f.write("### Weight Vector (w)\n\n")
        f.write("The weight vector defines the orientation of the decision boundary:\n\n")
        f.write(f"```\n")
        f.write(f"w = [{w[0]:.6f}, {w[1]:.6f}]\n")
        f.write(f"```\n\n")
        f.write(f"- **w₁** ({feature_names[0]}): {w[0]:.6f}\n")
        f.write(f"- **w₂** ({feature_names[1]}): {w[1]:.6f}\n\n")

        f.write("### Hyperplane Equation\n\n")
        f.write("The decision boundary is defined by the equation:\n\n")
        f.write(f"```\n")
        f.write(f"{w[0]:.6f} × {feature_names[0]} + {w[1]:.6f} × {feature_names[1]} + {b:.6f} = 0\n")
        f.write(f"```\n\n")

        f.write("### Geometric Properties\n\n")
        f.write(f"- **Bias (b)**: {b:.6f}\n")
        f.write(f"- **||w|| (Weight vector magnitude)**: {w_norm:.6f}\n")
        f.write(f"- **Margin**: {margin:.6f} (distance = 2/||w||)\n\n")

        f.write("#### Interpretation\n\n")
        f.write("- The **weight vector w** is perpendicular to the decision boundary hyperplane\n")
        f.write("- The direction of w points from the negative class (Setosa) toward the positive class (Non-Setosa)\n")
        f.write("- The **margin** represents the perpendicular distance between the hyperplane and the nearest support vectors\n")
        f.write("- A larger margin indicates better separation between classes\n\n")

        f.write("## Support Vectors\n\n")
        f.write(f"- **Total Support Vectors**: {len(support_vectors)}\n")
        f.write(f"- **Support Vectors for {class_names[0]}**: {n_support[0]}\n")
        f.write(f"- **Support Vectors for {class_names[1]}**: {n_support[1]}\n\n")

        f.write("Support vectors are the critical data points that:\n")
        f.write("- Lie on or within the margin boundaries\n")
        f.write("- Directly influence the position and orientation of the decision boundary\n")
        f.write("- Define the maximum margin hyperplane\n\n")

        f.write("## Model Performance\n\n")
        f.write("### Accuracy Scores\n\n")
        f.write(f"- **Training Accuracy**: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)\n")
        f.write(f"- **Testing Accuracy**: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n\n")

        f.write("### Confusion Matrix\n\n")
        f.write("```\n")
        f.write(f"                     Predicted {class_names[0]:<15} Predicted {class_names[1]}\n")
        f.write(f"Actual {class_names[0]:<15} {cm[0][0]:<32} {cm[0][1]}\n")
        f.write(f"Actual {class_names[1]:<15} {cm[1][0]:<32} {cm[1][1]}\n")
        f.write("```\n\n")

        f.write("### Classification Report\n\n")
        f.write("```\n")
        f.write(report)
        f.write("\n```\n\n")

        f.write("## Visualizations\n\n")
        f.write("### Main Visualization\n\n")
        f.write("![SVM Grouped Visualization](svm_visualization_grouped.png)\n\n")

        f.write("**Interpretation:**\n")
        f.write("- **Left Plot**: Shows the decision boundary (solid line), margin boundaries (dashed lines), ")
        f.write("support vectors (green circles), and the weight vector w (purple arrow)\n")
        f.write("- **Middle Plot**: Original three-class data with support vectors highlighted\n")
        f.write("- **Right Plot**: Detailed breakdown of hyperplane parameters and weight vector components\n\n")

        f.write("### Weight Vector Analysis\n\n")
        f.write("![Weight Vector Analysis](weight_vector_analysis.png)\n\n")

        f.write("## Mathematical Details\n\n")
        f.write("### SVM Decision Function\n\n")
        f.write("For a linear SVM, the decision function is:\n\n")
        f.write("```\n")
        f.write("f(x) = w · x + b\n")
        f.write("     = w₁×x₁ + w₂×x₂ + b\n")
        f.write("```\n\n")
        f.write("Where:\n")
        f.write("- **w** = weight vector (normal to the hyperplane)\n")
        f.write("- **x** = feature vector [x₁, x₂]\n")
        f.write("- **b** = bias term (intercept)\n\n")

        f.write("### Classification Rule\n\n")
        f.write("```\n")
        f.write(f"If f(x) ≥ 0: predict '{class_names[1]}'\n")
        f.write(f"If f(x) < 0: predict '{class_names[0]}'\n")
        f.write("```\n\n")

        f.write("### Margin Calculation\n\n")
        f.write("The margin between the two support vector boundaries is:\n\n")
        f.write("```\n")
        f.write("Margin = 2 / ||w||\n")
        f.write(f"       = 2 / {w_norm:.6f}\n")
        f.write(f"       = {margin:.6f}\n")
        f.write("```\n\n")

        f.write("## Key Findings\n\n")
        if test_accuracy >= 0.95:
            f.write("- The SVM model achieved **excellent** classification performance.\n")
        elif test_accuracy >= 0.85:
            f.write("- The SVM model achieved **good** classification performance.\n")
        else:
            f.write("- The SVM model achieved **moderate** classification performance.\n")

        f.write(f"- The decision boundary is defined by **{len(support_vectors)} support vectors** ")
        f.write(f"({len(support_vectors)/len(X_train)*100:.1f}% of training data).\n")
        f.write("- Linear kernel is sufficient to separate Setosa from Non-Setosa classes.\n")
        f.write(f"- The weight vector w = [{w[0]:.4f}, {w[1]:.4f}] indicates that ")

        # Determine which feature is more important
        if abs(w[0]) > abs(w[1]):
            f.write(f"**{feature_names[0]}** has a stronger influence on the classification decision.\n")
        else:
            f.write(f"**{feature_names[1]}** has a stronger influence on the classification decision.\n")

        if train_accuracy - test_accuracy > 0.1:
            f.write("- Note: Some overfitting detected (training accuracy significantly higher than testing).\n")
        else:
            f.write("- The model generalizes well to unseen data.\n")

        f.write("\n## Conclusion\n\n")
        f.write("By grouping Versicolor and Virginica together, we created a binary classification problem ")
        f.write("that allows for clear visualization and interpretation of the SVM's weight vector and decision boundary. ")
        f.write("The linear kernel provides an explicit mathematical representation of how the model separates the classes ")
        f.write("based on the two features (Petal Length and Petal Width).\n")

    print(f"Results saved to '{output_file}'")

def main():
    """Main function to run the grouped SVM classification simulation"""
    print("=" * 70)
    print("Binary Classification with Grouped Classes")
    print("SVM Weight Vector Analysis")
    print("=" * 70)
    print()

    # Load and prepare data
    print("Loading Iris dataset with grouped classes...")
    X, y, feature_names, class_names, original_classes = load_and_prepare_data()
    print(f"Dataset loaded: {len(X)} samples, 2 features")
    print(f"Original classes: {', '.join(original_classes)}")
    print(f"Grouped classes: {class_names[0]} vs {class_names[1]}")
    print(f"  - Class 0 ({class_names[0]}): {sum(y == 0)} samples")
    print(f"  - Class 1 ({class_names[1]}): {sum(y == 1)} samples")
    print()

    # Split data
    print("Splitting data into train/test sets (70/30)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    print()

    # Train model
    print("Training SVM model with linear kernel...")
    svm_model, scaler = train_svm_model(X_train, y_train)
    print(f"Model trained successfully!")
    print()

    # Calculate hyperplane parameters
    print("Calculating hyperplane parameters...")
    w, b, margin, w_norm = calculate_hyperplane_parameters(svm_model)
    print(f"Weight vector w: [{w[0]:.6f}, {w[1]:.6f}]")
    print(f"Bias b: {b:.6f}")
    print(f"||w||: {w_norm:.6f}")
    print(f"Margin: {margin:.6f}")
    print(f"Number of support vectors: {len(svm_model.support_vectors_)}")
    print(f"Support vectors per class: {svm_model.n_support_}")
    print()

    # Evaluate model
    print("Evaluating model performance...")
    X_test_scaled = scaler.transform(X_test)
    y_pred = svm_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print()

    # Generate visualizations
    print("Generating decision boundary visualization with weight vector...")
    plot_decision_boundary_with_hyperplane(X, y, svm_model, scaler,
                                          feature_names, class_names, original_classes)
    print()

    print("Generating weight vector analysis...")
    plot_weight_vector_analysis(svm_model, feature_names)
    print()

    # Generate results markdown
    print("Generating Results_grouped.md file...")
    generate_results_markdown(X_train, X_test, y_train, y_test,
                             svm_model, scaler, feature_names, class_names, original_classes)
    print()

    print("=" * 70)
    print("Simulation completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
