#!/usr/bin/env python3
"""
Main execution script for 3-class SVM classification.
Trains an SVM classifier on the Iris dataset using all 3 classes.
"""

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, confusion_matrix
from svm_3classes_data import load_and_prepare_data, split_data
from svm_3classes_model import train_svm_model
from svm_3classes_plotting import plot_decision_boundary, plot_confusion_matrix_heatmap
from svm_3classes_results import generate_results_markdown


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
    X_train, X_test, y_train, y_test = split_data(X, y)
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
