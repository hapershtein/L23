#!/usr/bin/env python3
"""
Main execution script for grouped SVM classification.
Groups Versicolor and Virginica into one class and classifies against Setosa.
"""

import warnings
warnings.filterwarnings('ignore')

from svm_grouped_data import load_and_prepare_data, split_data
from svm_grouped_model import train_svm_model, calculate_hyperplane_parameters, evaluate_model
from svm_grouped_plotting import plot_decision_boundary_with_hyperplane
from svm_grouped_weight_analysis import plot_weight_vector_analysis
from svm_grouped_results import generate_results_markdown


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
    X_train, X_test, y_train, y_test = split_data(X, y)
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
    test_accuracy, y_pred = evaluate_model(X_test, y_test, svm_model, scaler)
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
