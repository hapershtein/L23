#!/usr/bin/env python3
"""
Data loading and preparation for grouped SVM classification.
Groups Versicolor and Virginica into one class for binary classification.
"""

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


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


def split_data(X, y, test_size=0.3, random_state=42):
    """Split data into training and testing sets"""
    return train_test_split(X, y, test_size=test_size,
                           random_state=random_state, stratify=y)


def get_class_distribution(y):
    """Get the distribution of samples across classes"""
    unique, counts = np.unique(y, return_counts=True)
    return dict(zip(unique, counts))
