#!/usr/bin/env python3
"""
Binary Classification using Support Vector Machine on Iris Dataset
This script trains an SVM classifier on the Iris dataset (using only 2 classes)
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
    """Load Iris dataset and prepare for binary classification"""
    # Load iris dataset
    iris = datasets.load_iris()

    # Use only 2 features (sepal length and sepal width) for 2D visualization
    # Use only 2 classes (setosa=0 and versicolor=1) for binary classification
    X = iris.data[:100, :2]  # First 100 samples, first 2 features
    y = iris.target[:100]     # First 100 labels (0s and 1s)

    feature_names = ['Sepal Length (cm)', 'Sepal Width (cm)']
    class_names = ['Setosa', 'Versicolor']

    return X, y, feature_names, class_names

def train_svm_model(X_train, y_train):
    """Train SVM model with linear kernel"""
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train SVM with linear kernel
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(X_train_scaled, y_train)

    return svm_model, scaler

def plot_decision_boundary(X, y, model, scaler, feature_names, class_names, filename='svm_visualization.png'):
    """Plot decision boundary, support vectors, and data points"""
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
    plt.figure(figsize=(12, 5))

    # Plot 1: Decision boundary with support vectors
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5, linestyles='dashed')

    # Plot the training points
    scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y,
                         cmap=plt.cm.RdYlBu, edgecolors='k', s=100)

    # Plot support vectors
    plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                s=200, linewidth=2, facecolors='none', edgecolors='green',
                label='Support Vectors')

    plt.xlabel(f'{feature_names[0]} (scaled)', fontsize=12)
    plt.ylabel(f'{feature_names[1]} (scaled)', fontsize=12)
    plt.title('SVM Decision Boundary with Support Vectors', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Original data with support vectors marked
    plt.subplot(1, 2, 2)
    colors = ['red', 'blue']
    for idx, class_name in enumerate(class_names):
        plt.scatter(X[y == idx, 0], X[y == idx, 1],
                   c=colors[idx], label=class_name,
                   edgecolors='k', s=100, alpha=0.7)

    # Mark support vectors
    support_vector_indices = model.support_
    plt.scatter(X[support_vector_indices, 0], X[support_vector_indices, 1],
                s=200, linewidth=2, facecolors='none', edgecolors='green',
                label=f'Support Vectors (n={len(support_vector_indices)})')

    plt.xlabel(feature_names[0], fontsize=12)
    plt.ylabel(feature_names[1], fontsize=12)
    plt.title('Original Data with Support Vectors', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as '{filename}'")
    plt.close()

def generate_results_markdown(X_train, X_test, y_train, y_test, model, scaler,
                              feature_names, class_names, output_file='Results.md'):
    """Generate markdown file with results"""
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
        f.write("# Binary Classification using Support Vector Machine\n\n")
        f.write("## Dataset Information\n\n")
        f.write("- **Dataset**: Iris Dataset (Binary Classification)\n")
        f.write(f"- **Classes**: {class_names[0]} vs {class_names[1]}\n")
        f.write(f"- **Features Used**: {feature_names[0]}, {feature_names[1]}\n")
        f.write(f"- **Total Samples**: {len(X_train) + len(X_test)}\n")
        f.write(f"- **Training Samples**: {len(X_train)}\n")
        f.write(f"- **Testing Samples**: {len(X_test)}\n\n")

        f.write("## SVM Model Configuration\n\n")
        f.write("- **Kernel**: Linear\n")
        f.write("- **C Parameter**: 1.0\n")
        f.write("- **Feature Scaling**: StandardScaler\n\n")

        f.write("## Support Vectors\n\n")
        f.write(f"- **Total Support Vectors**: {len(support_vectors)}\n")
        f.write(f"- **Support Vectors for {class_names[0]}**: {n_support[0]}\n")
        f.write(f"- **Support Vectors for {class_names[1]}**: {n_support[1]}\n\n")

        f.write("## Model Performance\n\n")
        f.write("### Accuracy Scores\n\n")
        f.write(f"- **Training Accuracy**: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)\n")
        f.write(f"- **Testing Accuracy**: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n\n")

        f.write("### Confusion Matrix\n\n")
        f.write("```\n")
        f.write(f"                Predicted {class_names[0]}  Predicted {class_names[1]}\n")
        f.write(f"Actual {class_names[0]:<12} {cm[0][0]:<19} {cm[0][1]}\n")
        f.write(f"Actual {class_names[1]:<12} {cm[1][0]:<19} {cm[1][1]}\n")
        f.write("```\n\n")

        f.write("### Classification Report\n\n")
        f.write("```\n")
        f.write(report)
        f.write("\n```\n\n")

        f.write("## Visualization\n\n")
        f.write("![SVM Visualization](svm_visualization.png)\n\n")
        f.write("### Interpretation\n\n")
        f.write("- **Left Plot**: Shows the decision boundary (background colors) and support vectors (green circles) in scaled feature space.\n")
        f.write("- **Right Plot**: Shows the original data points with support vectors highlighted.\n")
        f.write("- **Support Vectors**: These are the critical data points that define the decision boundary.\n")
        f.write("- **Decision Boundary**: The line that separates the two classes, positioned to maximize the margin between them.\n\n")

        f.write("## Key Findings\n\n")
        if test_accuracy >= 0.95:
            f.write("- The SVM model achieved excellent classification performance.\n")
        elif test_accuracy >= 0.85:
            f.write("- The SVM model achieved good classification performance.\n")
        else:
            f.write("- The SVM model achieved moderate classification performance.\n")

        f.write(f"- The decision boundary is defined by {len(support_vectors)} support vectors.\n")
        f.write("- Linear kernel was sufficient to separate the two classes effectively.\n")

        if train_accuracy - test_accuracy > 0.1:
            f.write("- Note: Some overfitting detected (training accuracy significantly higher than testing).\n")
        else:
            f.write("- The model generalizes well to unseen data.\n")

    print(f"Results saved to '{output_file}'")

def main():
    """Main function to run the SVM classification simulation"""
    print("=" * 60)
    print("Binary Classification with Support Vector Machine")
    print("=" * 60)
    print()

    # Load and prepare data
    print("Loading Iris dataset...")
    X, y, feature_names, class_names = load_and_prepare_data()
    print(f"Dataset loaded: {len(X)} samples, 2 features, 2 classes")
    print(f"Classes: {class_names[0]} vs {class_names[1]}")
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
    print(f"Number of support vectors: {len(svm_model.support_vectors_)}")
    print()

    # Evaluate model
    print("Evaluating model performance...")
    X_test_scaled = scaler.transform(X_test)
    y_pred = svm_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print()

    # Generate visualization
    print("Generating visualization...")
    plot_decision_boundary(X, y, svm_model, scaler, feature_names, class_names)
    print()

    # Generate results markdown
    print("Generating Results.md file...")
    generate_results_markdown(X_train, X_test, y_train, y_test,
                             svm_model, scaler, feature_names, class_names)
    print()

    print("=" * 60)
    print("Simulation completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
