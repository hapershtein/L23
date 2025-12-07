#!/usr/bin/env python3
"""
Data loading and preparation for binary SVM classification.
Uses 2 classes and 2 features from the Iris dataset.
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split


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


def split_data(X, y, test_size=0.3, random_state=42):
    """Split data into training and testing sets"""
    return train_test_split(X, y, test_size=test_size,
                           random_state=random_state, stratify=y)
