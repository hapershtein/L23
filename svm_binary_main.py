#!/usr/bin/env python3
"""
Main execution script for binary SVM classification.
Trains an SVM classifier on 2 classes from the Iris dataset.
"""

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score
from svm_binary_data import load_and_prepare_data, split_data
from svm_binary_model import train_svm_model
from svm_binary_plotting import plot_decision_boundary
from svm_binary_results import generate_results_markdown


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
    X_train, X_test, y_train, y_test = split_data(X, y)
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
