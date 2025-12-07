#!/usr/bin/env python3
"""
SVM model training and hyperplane parameter calculations for grouped classification.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


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


def evaluate_model(X_test, y_test, model, scaler):
    """Evaluate model performance on test data"""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy, y_pred


def get_predictions(X, model, scaler):
    """Get predictions for given data"""
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)


def get_model_metrics(y_true, y_pred, class_names):
    """Calculate various performance metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names)

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }
