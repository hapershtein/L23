#!/usr/bin/env python3
"""
Multi-class Classification using Support Vector Machine on Iris Dataset
This script trains an SVM classifier on the Iris dataset (using all 3 classes)
and generates visualizations and results.
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
    """Load Iris dataset with all 3 classes"""
    # Load iris dataset
    iris = datasets.load_iris()

    # Use only 2 features (petal length and petal width) for 2D visualization
    # Use all 3 classes (setosa=0, versicolor=1, virginica=2)
    X = iris.data[:, 2:4]  # Petal length and petal width (better separation for 3 classes)
    y = iris.target        # All 150 samples with 3 classes

    feature_names = ['Petal Length (cm)', 'Petal Width (cm)']
    class_names = ['Setosa', 'Versicolor', 'Virginica']

    return X, y, feature_names, class_names

def train_svm_model(X_train, y_train):
    """Train SVM model with RBF kernel for multi-class classification"""
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train SVM with RBF kernel (better for non-linear boundaries)
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, decision_function_shape='ovr')
    svm_model.fit(X_train_scaled, y_train)

    return svm_model, scaler

def plot_decision_boundary(X, y, model, scaler, feature_names, class_names, filename='svm_visualization_3classes.png'):
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

    # Plot 2: Original data with support vectors marked
    plt.subplot(1, 3, 2)
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

    # Plot 3: Support vectors distribution by class
    plt.subplot(1, 3, 3)
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

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as '{filename}'")
    plt.close()

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

def generate_results_markdown(X_train, X_test, y_train, y_test, model, scaler,
                              feature_names, class_names, output_file='Results_3_classes.md'):
    """Generate markdown file with results for 3-class classification"""
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

    # Write to markdown file
    with open(output_file, 'w') as f:
        f.write("# Multi-class Classification using Support Vector Machine\n\n")
        f.write("## Dataset Information\n\n")
        f.write("- **Dataset**: Iris Dataset (3-Class Classification)\n")
        f.write(f"- **Classes**: {', '.join(class_names)}\n")
        f.write(f"- **Features Used**: {feature_names[0]}, {feature_names[1]}\n")
        f.write(f"- **Total Samples**: {len(X_train) + len(X_test)}\n")
        f.write(f"- **Training Samples**: {len(X_train)}\n")
        f.write(f"- **Testing Samples**: {len(X_test)}\n")
        f.write(f"- **Samples per Class**: 50 each (balanced dataset)\n\n")

        f.write("## SVM Model Configuration\n\n")
        f.write("- **Kernel**: RBF (Radial Basis Function)\n")
        f.write("- **C Parameter**: 1.0\n")
        f.write("- **Gamma**: scale (auto-computed)\n")
        f.write("- **Decision Function**: One-vs-Rest (OvR)\n")
        f.write("- **Feature Scaling**: StandardScaler\n\n")

        f.write("### Multi-class Strategy\n\n")
        f.write("The SVM uses a **One-vs-Rest (OvR)** approach for multi-class classification:\n")
        f.write("- Trains 3 binary classifiers (one for each class vs. the rest)\n")
        f.write("- Each classifier learns to distinguish one class from all others\n")
        f.write("- Final prediction is made by choosing the class with highest confidence\n\n")

        f.write("## Support Vectors\n\n")
        f.write(f"- **Total Support Vectors**: {len(support_vectors)}\n")
        for idx, class_name in enumerate(class_names):
            f.write(f"- **Support Vectors for {class_name}**: {n_support[idx]}\n")
        f.write(f"\n**Note**: {len(support_vectors)} out of {len(X_train)} training samples ")
        f.write(f"({len(support_vectors)/len(X_train)*100:.1f}%) are support vectors.\n\n")

        f.write("## Model Performance\n\n")
        f.write("### Accuracy Scores\n\n")
        f.write(f"- **Training Accuracy**: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)\n")
        f.write(f"- **Testing Accuracy**: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n\n")

        f.write("### Confusion Matrix\n\n")
        f.write("```\n")
        f.write("                  Predicted    Predicted      Predicted\n")
        f.write(f"                  {class_names[0]:<12} {class_names[1]:<14} {class_names[2]}\n")
        for idx, class_name in enumerate(class_names):
            f.write(f"Actual {class_name:<12} {cm[idx][0]:<13} {cm[idx][1]:<15} {cm[idx][2]}\n")
        f.write("```\n\n")

        f.write("### Classification Report\n\n")
        f.write("```\n")
        f.write(report)
        f.write("\n```\n\n")

        f.write("## Visualizations\n\n")
        f.write("### Decision Boundaries and Support Vectors\n\n")
        f.write("![SVM Visualization 3 Classes](svm_visualization_3classes.png)\n\n")

        f.write("### Confusion Matrix Heatmap\n\n")
        f.write("![Confusion Matrix Heatmap](confusion_matrix_3classes.png)\n\n")

        f.write("## Interpretation\n\n")
        f.write("### Decision Boundary Visualization\n\n")
        f.write("- **Left Plot**: Shows the decision boundaries (colored regions) and support vectors (black circles) ")
        f.write("in scaled feature space. Each color represents a different class region.\n")
        f.write("- **Middle Plot**: Shows the original data points with support vectors highlighted in black circles.\n")
        f.write("- **Right Plot**: Bar chart showing the distribution of support vectors across the three classes.\n\n")

        f.write("### Key Observations\n\n")
        f.write("- **Support Vectors**: These critical data points define the decision boundaries between classes.\n")
        f.write("- **RBF Kernel**: Allows for non-linear decision boundaries, creating more flexible separation between classes.\n")
        f.write("- **Multi-class Boundaries**: Multiple decision boundaries separate the three classes in feature space.\n\n")

        f.write("## Detailed Analysis\n\n")

        # Per-class analysis
        f.write("### Per-Class Performance\n\n")
        for idx, class_name in enumerate(class_names):
            true_positives = cm[idx][idx]
            total_actual = sum(cm[idx])
            total_predicted = sum(cm[i][idx] for i in range(len(class_names)))

            if total_actual > 0:
                recall = true_positives / total_actual
                precision = true_positives / total_predicted if total_predicted > 0 else 0

                f.write(f"#### {class_name}\n")
                f.write(f"- **Precision**: {precision:.4f} - Of all samples predicted as {class_name}, ")
                f.write(f"{precision*100:.1f}% were correct\n")
                f.write(f"- **Recall**: {recall:.4f} - Of all actual {class_name} samples, ")
                f.write(f"{recall*100:.1f}% were correctly identified\n")
                f.write(f"- **Support Vectors**: {n_support[idx]} points from this class are critical for defining boundaries\n\n")

        f.write("## Key Findings\n\n")
        if test_accuracy >= 0.95:
            f.write("- The SVM model achieved **excellent** classification performance on all three classes.\n")
        elif test_accuracy >= 0.85:
            f.write("- The SVM model achieved **good** classification performance on all three classes.\n")
        else:
            f.write("- The SVM model achieved **moderate** classification performance on all three classes.\n")

        f.write(f"- The decision boundaries are defined by {len(support_vectors)} support vectors ")
        f.write(f"({len(support_vectors)/len(X_train)*100:.1f}% of training data).\n")
        f.write("- RBF kernel successfully handles the non-linear separation between the three Iris species.\n")

        if train_accuracy - test_accuracy > 0.1:
            f.write("- **Note**: Some overfitting detected (training accuracy significantly higher than testing).\n")
        else:
            f.write("- The model generalizes well to unseen data across all three classes.\n")

        # Identify most confused classes
        max_confusion = 0
        confused_classes = None
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i][j] > max_confusion:
                    max_confusion = cm[i][j]
                    confused_classes = (class_names[i], class_names[j])

        if max_confusion > 0 and confused_classes:
            f.write(f"- Most confusion occurs between **{confused_classes[0]}** and **{confused_classes[1]}** ")
            f.write(f"({max_confusion} misclassifications).\n")
        else:
            f.write("- Perfect separation achieved with no misclassifications.\n")

    print(f"Results saved to '{output_file}'")

def main():
    """Main function to run the 3-class SVM classification simulation"""
    print("=" * 60)
    print("Multi-class Classification with Support Vector Machine")
    print("3 Classes: Setosa, Versicolor, Virginica")
    print("=" * 60)
    print()

    # Load and prepare data
    print("Loading Iris dataset (all 3 classes)...")
    X, y, feature_names, class_names = load_and_prepare_data()
    print(f"Dataset loaded: {len(X)} samples, 2 features, 3 classes")
    print(f"Classes: {', '.join(class_names)}")
    print()

    # Split data
    print("Splitting data into train/test sets (70/30)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    print()

    # Train model
    print("Training SVM model with RBF kernel...")
    svm_model, scaler = train_svm_model(X_train, y_train)
    print(f"Model trained successfully!")
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

    # Generate confusion matrix heatmap
    print("Generating confusion matrix heatmap...")
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix_heatmap(cm, class_names)
    print()

    # Generate visualization
    print("Generating decision boundary visualization...")
    plot_decision_boundary(X, y, svm_model, scaler, feature_names, class_names)
    print()

    # Generate results markdown
    print("Generating Results_3_classes.md file...")
    generate_results_markdown(X_train, X_test, y_train, y_test,
                             svm_model, scaler, feature_names, class_names)
    print()

    print("=" * 60)
    print("Simulation completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
