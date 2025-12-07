#!/usr/bin/env python3
"""
Data loading and preparation for 3-class SVM classification.
Handles loading the Iris dataset with all 3 classes.
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split


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


def split_data(X, y, test_size=0.3, random_state=42):
    """Split data into training and testing sets"""
    return train_test_split(X, y, test_size=test_size,
                           random_state=random_state, stratify=y)
