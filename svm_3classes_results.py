#!/usr/bin/env python3
"""
Results generation and markdown export for 3-class SVM classification.
"""

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def generate_results_markdown(X_train, X_test, y_train, y_test, model, scaler,
                              feature_names, class_names, output_file='Results_3_classes.md'):
    """Generate markdown file with results for 3-class classification"""
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
        _write_header_section(f, feature_names, class_names, X_train, X_test)
        _write_model_configuration(f)
        _write_support_vectors_section(f, support_vectors, n_support, class_names, X_train)
        _write_performance_section(f, train_accuracy, test_accuracy, cm, report, class_names)
        _write_visualizations_section(f)
        _write_interpretation_section(f)
        _write_detailed_analysis(f, cm, n_support, class_names)
        _write_key_findings(f, test_accuracy, train_accuracy, support_vectors, X_train, cm, class_names)

    print(f"Results saved to '{output_file}'")


def _write_header_section(f, feature_names, class_names, X_train, X_test):
    """Write header and dataset information"""
    f.write("# Multi-class Classification using Support Vector Machine\n\n")
    f.write("## Dataset Information\n\n")
    f.write("- **Dataset**: Iris Dataset (3-Class Classification)\n")
    f.write(f"- **Classes**: {', '.join(class_names)}\n")
    f.write(f"- **Features Used**: {feature_names[0]}, {feature_names[1]}\n")
    f.write(f"- **Total Samples**: {len(X_train) + len(X_test)}\n")
    f.write(f"- **Training Samples**: {len(X_train)}\n")
    f.write(f"- **Testing Samples**: {len(X_test)}\n")
    f.write(f"- **Samples per Class**: 50 each (balanced dataset)\n\n")


def _write_model_configuration(f):
    """Write SVM model configuration details"""
    f.write("## SVM Model Configuration\n\n")
    f.write("- **Kernel**: RBF (Radial Basis Function)\n")
    f.write("- **C Parameter**: 1.0\n")
    f.write("- **Gamma**: scale (auto-computed)\n")
    f.write("- **Decision Function**: One-vs-Rest (OvR)\n")
    f.write("- **Feature Scaling**: StandardScaler\n\n")

    f.write("### Multi-class Strategy\n\n")
    f.write("The SVM uses a **One-vs-Rest (OvR)** approach for multi-class classification:\n")
    f.write("- Trains 3 binary classifiers (one for each class vs. the rest)\n")
    f.write("- Each classifier learns to distinguish one class from all others\n")
    f.write("- Final prediction is made by choosing the class with highest confidence\n\n")


def _write_support_vectors_section(f, support_vectors, n_support, class_names, X_train):
    """Write support vectors information"""
    f.write("## Support Vectors\n\n")
    f.write(f"- **Total Support Vectors**: {len(support_vectors)}\n")
    for idx, class_name in enumerate(class_names):
        f.write(f"- **Support Vectors for {class_name}**: {n_support[idx]}\n")
    f.write(f"\n**Note**: {len(support_vectors)} out of {len(X_train)} training samples ")
    f.write(f"({len(support_vectors)/len(X_train)*100:.1f}%) are support vectors.\n\n")


def _write_performance_section(f, train_accuracy, test_accuracy, cm, report, class_names):
    """Write model performance metrics"""
    f.write("## Model Performance\n\n")
    f.write("### Accuracy Scores\n\n")
    f.write(f"- **Training Accuracy**: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)\n")
    f.write(f"- **Testing Accuracy**: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n\n")

    f.write("### Confusion Matrix\n\n")
    f.write("```\n")
    f.write("                  Predicted    Predicted      Predicted\n")
    f.write(f"                  {class_names[0]:<12} {class_names[1]:<14} {class_names[2]}\n")
    for idx, class_name in enumerate(class_names):
        f.write(f"Actual {class_name:<12} {cm[idx][0]:<13} {cm[idx][1]:<15} {cm[idx][2]}\n")
    f.write("```\n\n")

    f.write("### Classification Report\n\n")
    f.write("```\n")
    f.write(report)
    f.write("\n```\n\n")


def _write_visualizations_section(f):
    """Write visualizations section"""
    f.write("## Visualizations\n\n")
    f.write("### Decision Boundaries and Support Vectors\n\n")
    f.write("![SVM Visualization 3 Classes](svm_visualization_3classes.png)\n\n")

    f.write("### Confusion Matrix Heatmap\n\n")
    f.write("![Confusion Matrix Heatmap](confusion_matrix_3classes.png)\n\n")


def _write_interpretation_section(f):
    """Write interpretation of visualizations"""
    f.write("## Interpretation\n\n")
    f.write("### Decision Boundary Visualization\n\n")
    f.write("- **Left Plot**: Shows the decision boundaries (colored regions) and support vectors (black circles) ")
    f.write("in scaled feature space. Each color represents a different class region.\n")
    f.write("- **Middle Plot**: Shows the original data points with support vectors highlighted in black circles.\n")
    f.write("- **Right Plot**: Bar chart showing the distribution of support vectors across the three classes.\n\n")

    f.write("### Key Observations\n\n")
    f.write("- **Support Vectors**: These critical data points define the decision boundaries between classes.\n")
    f.write("- **RBF Kernel**: Allows for non-linear decision boundaries, creating more flexible separation between classes.\n")
    f.write("- **Multi-class Boundaries**: Multiple decision boundaries separate the three classes in feature space.\n\n")


def _write_detailed_analysis(f, cm, n_support, class_names):
    """Write detailed per-class analysis"""
    f.write("## Detailed Analysis\n\n")
    f.write("### Per-Class Performance\n\n")
    for idx, class_name in enumerate(class_names):
        true_positives = cm[idx][idx]
        total_actual = sum(cm[idx])
        total_predicted = sum(cm[i][idx] for i in range(len(class_names)))

        if total_actual > 0:
            recall = true_positives / total_actual
            precision = true_positives / total_predicted if total_predicted > 0 else 0

            f.write(f"#### {class_name}\n")
            f.write(f"- **Precision**: {precision:.4f} - Of all samples predicted as {class_name}, ")
            f.write(f"{precision*100:.1f}% were correct\n")
            f.write(f"- **Recall**: {recall:.4f} - Of all actual {class_name} samples, ")
            f.write(f"{recall*100:.1f}% were correctly identified\n")
            f.write(f"- **Support Vectors**: {n_support[idx]} points from this class are critical for defining boundaries\n\n")


def _write_key_findings(f, test_accuracy, train_accuracy, support_vectors, X_train, cm, class_names):
    """Write key findings and conclusions"""
    f.write("## Key Findings\n\n")
    if test_accuracy >= 0.95:
        f.write("- The SVM model achieved **excellent** classification performance on all three classes.\n")
    elif test_accuracy >= 0.85:
        f.write("- The SVM model achieved **good** classification performance on all three classes.\n")
    else:
        f.write("- The SVM model achieved **moderate** classification performance on all three classes.\n")

    f.write(f"- The decision boundaries are defined by {len(support_vectors)} support vectors ")
    f.write(f"({len(support_vectors)/len(X_train)*100:.1f}% of training data).\n")
    f.write("- RBF kernel successfully handles the non-linear separation between the three Iris species.\n")

    if train_accuracy - test_accuracy > 0.1:
        f.write("- **Note**: Some overfitting detected (training accuracy significantly higher than testing).\n")
    else:
        f.write("- The model generalizes well to unseen data across all three classes.\n")

    # Identify most confused classes
    max_confusion = 0
    confused_classes = None
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i][j] > max_confusion:
                max_confusion = cm[i][j]
                confused_classes = (class_names[i], class_names[j])

    if max_confusion > 0 and confused_classes:
        f.write(f"- Most confusion occurs between **{confused_classes[0]}** and **{confused_classes[1]}** ")
        f.write(f"({max_confusion} misclassifications).\n")
    else:
        f.write("- Perfect separation achieved with no misclassifications.\n")
