#!/usr/bin/env python3
"""
Results generation and markdown export for binary SVM classification.
"""

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


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
        _write_header(f, class_names, feature_names, X_train, X_test)
        _write_model_config(f)
        _write_support_vectors(f, support_vectors, n_support, class_names)
        _write_performance(f, train_accuracy, test_accuracy, cm, report, class_names)
        _write_visualization(f)
        _write_key_findings(f, test_accuracy, train_accuracy, support_vectors)

    print(f"Results saved to '{output_file}'")


def _write_header(f, class_names, feature_names, X_train, X_test):
    """Write header and dataset information"""
    f.write("# Binary Classification using Support Vector Machine\n\n")
    f.write("## Dataset Information\n\n")
    f.write("- **Dataset**: Iris Dataset (Binary Classification)\n")
    f.write(f"- **Classes**: {class_names[0]} vs {class_names[1]}\n")
    f.write(f"- **Features Used**: {feature_names[0]}, {feature_names[1]}\n")
    f.write(f"- **Total Samples**: {len(X_train) + len(X_test)}\n")
    f.write(f"- **Training Samples**: {len(X_train)}\n")
    f.write(f"- **Testing Samples**: {len(X_test)}\n\n")


def _write_model_config(f):
    """Write model configuration section"""
    f.write("## SVM Model Configuration\n\n")
    f.write("- **Kernel**: Linear\n")
    f.write("- **C Parameter**: 1.0\n")
    f.write("- **Feature Scaling**: StandardScaler\n\n")


def _write_support_vectors(f, support_vectors, n_support, class_names):
    """Write support vectors information"""
    f.write("## Support Vectors\n\n")
    f.write(f"- **Total Support Vectors**: {len(support_vectors)}\n")
    f.write(f"- **Support Vectors for {class_names[0]}**: {n_support[0]}\n")
    f.write(f"- **Support Vectors for {class_names[1]}**: {n_support[1]}\n\n")


def _write_performance(f, train_accuracy, test_accuracy, cm, report, class_names):
    """Write model performance section"""
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


def _write_visualization(f):
    """Write visualization section"""
    f.write("## Visualization\n\n")
    f.write("![SVM Visualization](svm_visualization.png)\n\n")
    f.write("### Interpretation\n\n")
    f.write("- **Left Plot**: Shows the decision boundary (background colors) and support vectors (green circles) in scaled feature space.\n")
    f.write("- **Right Plot**: Shows the original data points with support vectors highlighted.\n")
    f.write("- **Support Vectors**: These are the critical data points that define the decision boundary.\n")
    f.write("- **Decision Boundary**: The line that separates the two classes, positioned to maximize the margin between them.\n\n")


def _write_key_findings(f, test_accuracy, train_accuracy, support_vectors):
    """Write key findings section"""
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
