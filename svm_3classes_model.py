#!/usr/bin/env python3
"""
SVM model training and evaluation for 3-class classification.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def train_svm_model(X_train, y_train):
    """Train SVM model with RBF kernel for multi-class classification"""
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train SVM with RBF kernel (better for non-linear boundaries)
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, decision_function_shape='ovr')
    svm_model.fit(X_train_scaled, y_train)

    return svm_model, scaler


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
