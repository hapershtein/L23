#!/usr/bin/env python3
"""
Results generation and markdown export for grouped SVM classification.
"""

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from svm_grouped_model import calculate_hyperplane_parameters


def generate_results_markdown(X_train, X_test, y_train, y_test, model, scaler,
                              feature_names, class_names, original_classes,
                              output_file='Results_grouped.md'):
    """Generate markdown file with results for grouped classification"""
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

    # Get hyperplane parameters
    w, b, margin, w_norm = calculate_hyperplane_parameters(model)

    # Write to markdown file
    with open(output_file, 'w') as f:
        _write_overview_section(f, original_classes)
        _write_dataset_section(f, X_train, X_test, feature_names, class_names, original_classes)
        _write_hyperplane_section(f, w, b, margin, w_norm, feature_names)
        _write_support_vectors_section(f, support_vectors, n_support, class_names)
        _write_performance_section(f, train_accuracy, test_accuracy, cm, report, class_names)
        _write_visualizations_section(f)
        _write_mathematical_details(f, w, b, margin, w_norm, class_names)
        _write_findings_section(f, test_accuracy, train_accuracy, support_vectors,
                               X_train, w, feature_names)
        _write_conclusion_section(f)

    print(f"Results saved to '{output_file}'")


def _write_overview_section(f, original_classes):
    """Write overview section"""
    f.write("# Binary Classification with Grouped Classes using Support Vector Machine\n\n")
    f.write("## Overview\n\n")
    f.write("This analysis groups two of the three Iris classes together to perform binary classification:\n")
    f.write(f"- **Class 0**: {original_classes[0]}\n")
    f.write(f"- **Class 1**: {original_classes[1]} + {original_classes[2]} (grouped as 'Non-Setosa')\n\n")
    f.write("This grouping allows us to calculate and analyze the **weight vector (w)** and support vectors ")
    f.write("for a linear decision boundary.\n\n")


def _write_dataset_section(f, X_train, X_test, feature_names, class_names, original_classes):
    """Write dataset information section"""
    f.write("## Dataset Information\n\n")
    f.write("- **Dataset**: Iris Dataset (Binary Classification with Grouped Classes)\n")
    f.write(f"- **Original Classes**: {', '.join(original_classes)}\n")
    f.write(f"- **Grouped Classes**: {class_names[0]} vs {class_names[1]}\n")
    f.write(f"- **Features Used**: {feature_names[0]}, {feature_names[1]}\n")
    f.write(f"- **Total Samples**: {len(X_train) + len(X_test)}\n")
    f.write(f"- **Training Samples**: {len(X_train)}\n")
    f.write(f"- **Testing Samples**: {len(X_test)}\n\n")
    f.write("## SVM Model Configuration\n\n")
    f.write("- **Kernel**: Linear (enables calculation of explicit weight vector)\n")
    f.write("- **C Parameter**: 1.0\n")
    f.write("- **Feature Scaling**: StandardScaler\n\n")


def _write_hyperplane_section(f, w, b, margin, w_norm, feature_names):
    """Write hyperplane parameters section"""
    f.write("## Hyperplane Parameters\n\n")
    f.write("### Weight Vector (w)\n\n")
    f.write("The weight vector defines the orientation of the decision boundary:\n\n")
    f.write(f"```\n")
    f.write(f"w = [{w[0]:.6f}, {w[1]:.6f}]\n")
    f.write(f"```\n\n")
    f.write(f"- **w₁** ({feature_names[0]}): {w[0]:.6f}\n")
    f.write(f"- **w₂** ({feature_names[1]}): {w[1]:.6f}\n\n")

    f.write("### Hyperplane Equation\n\n")
    f.write("The decision boundary is defined by the equation:\n\n")
    f.write(f"```\n")
    f.write(f"{w[0]:.6f} × {feature_names[0]} + {w[1]:.6f} × {feature_names[1]} + {b:.6f} = 0\n")
    f.write(f"```\n\n")

    f.write("### Geometric Properties\n\n")
    f.write(f"- **Bias (b)**: {b:.6f}\n")
    f.write(f"- **||w|| (Weight vector magnitude)**: {w_norm:.6f}\n")
    f.write(f"- **Margin**: {margin:.6f} (distance = 2/||w||)\n\n")

    f.write("#### Interpretation\n\n")
    f.write("- The **weight vector w** is perpendicular to the decision boundary hyperplane\n")
    f.write("- The direction of w points from the negative class (Setosa) toward the positive class (Non-Setosa)\n")
    f.write("- The **margin** represents the perpendicular distance between the hyperplane and the nearest support vectors\n")
    f.write("- A larger margin indicates better separation between classes\n\n")


def _write_support_vectors_section(f, support_vectors, n_support, class_names):
    """Write support vectors section"""
    f.write("## Support Vectors\n\n")
    f.write(f"- **Total Support Vectors**: {len(support_vectors)}\n")
    f.write(f"- **Support Vectors for {class_names[0]}**: {n_support[0]}\n")
    f.write(f"- **Support Vectors for {class_names[1]}**: {n_support[1]}\n\n")

    f.write("Support vectors are the critical data points that:\n")
    f.write("- Lie on or within the margin boundaries\n")
    f.write("- Directly influence the position and orientation of the decision boundary\n")
    f.write("- Define the maximum margin hyperplane\n\n")


def _write_performance_section(f, train_accuracy, test_accuracy, cm, report, class_names):
    """Write model performance section"""
    f.write("## Model Performance\n\n")
    f.write("### Accuracy Scores\n\n")
    f.write(f"- **Training Accuracy**: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)\n")
    f.write(f"- **Testing Accuracy**: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)\n\n")

    f.write("### Confusion Matrix\n\n")
    f.write("```\n")
    f.write(f"                     Predicted {class_names[0]:<15} Predicted {class_names[1]}\n")
    f.write(f"Actual {class_names[0]:<15} {cm[0][0]:<32} {cm[0][1]}\n")
    f.write(f"Actual {class_names[1]:<15} {cm[1][0]:<32} {cm[1][1]}\n")
    f.write("```\n\n")

    f.write("### Classification Report\n\n")
    f.write("```\n")
    f.write(report)
    f.write("\n```\n\n")


def _write_visualizations_section(f):
    """Write visualizations section"""
    f.write("## Visualizations\n\n")
    f.write("### Main Visualization\n\n")
    f.write("![SVM Grouped Visualization](svm_visualization_grouped.png)\n\n")

    f.write("**Interpretation:**\n")
    f.write("- **Left Plot**: Shows the decision boundary (solid line), margin boundaries (dashed lines), ")
    f.write("support vectors (green circles), and the weight vector w (purple arrow)\n")
    f.write("- **Middle Plot**: Original three-class data with support vectors highlighted\n")
    f.write("- **Right Plot**: Detailed breakdown of hyperplane parameters and weight vector components\n\n")

    f.write("### Weight Vector Analysis\n\n")
    f.write("![Weight Vector Analysis](weight_vector_analysis.png)\n\n")


def _write_mathematical_details(f, w, b, margin, w_norm, class_names):
    """Write mathematical details section"""
    f.write("## Mathematical Details\n\n")
    f.write("### SVM Decision Function\n\n")
    f.write("For a linear SVM, the decision function is:\n\n")
    f.write("```\n")
    f.write("f(x) = w · x + b\n")
    f.write("     = w₁×x₁ + w₂×x₂ + b\n")
    f.write("```\n\n")
    f.write("Where:\n")
    f.write("- **w** = weight vector (normal to the hyperplane)\n")
    f.write("- **x** = feature vector [x₁, x₂]\n")
    f.write("- **b** = bias term (intercept)\n\n")

    f.write("### Classification Rule\n\n")
    f.write("```\n")
    f.write(f"If f(x) ≥ 0: predict '{class_names[1]}'\n")
    f.write(f"If f(x) < 0: predict '{class_names[0]}'\n")
    f.write("```\n\n")

    f.write("### Margin Calculation\n\n")
    f.write("The margin between the two support vector boundaries is:\n\n")
    f.write("```\n")
    f.write("Margin = 2 / ||w||\n")
    f.write(f"       = 2 / {w_norm:.6f}\n")
    f.write(f"       = {margin:.6f}\n")
    f.write("```\n\n")


def _write_findings_section(f, test_accuracy, train_accuracy, support_vectors,
                            X_train, w, feature_names):
    """Write key findings section"""
    f.write("## Key Findings\n\n")
    if test_accuracy >= 0.95:
        f.write("- The SVM model achieved **excellent** classification performance.\n")
    elif test_accuracy >= 0.85:
        f.write("- The SVM model achieved **good** classification performance.\n")
    else:
        f.write("- The SVM model achieved **moderate** classification performance.\n")

    f.write(f"- The decision boundary is defined by **{len(support_vectors)} support vectors** ")
    f.write(f"({len(support_vectors)/len(X_train)*100:.1f}% of training data).\n")
    f.write("- Linear kernel is sufficient to separate Setosa from Non-Setosa classes.\n")
    f.write(f"- The weight vector w = [{w[0]:.4f}, {w[1]:.4f}] indicates that ")

    # Determine which feature is more important
    if abs(w[0]) > abs(w[1]):
        f.write(f"**{feature_names[0]}** has a stronger influence on the classification decision.\n")
    else:
        f.write(f"**{feature_names[1]}** has a stronger influence on the classification decision.\n")

    if train_accuracy - test_accuracy > 0.1:
        f.write("- Note: Some overfitting detected (training accuracy significantly higher than testing).\n")
    else:
        f.write("- The model generalizes well to unseen data.\n")


def _write_conclusion_section(f):
    """Write conclusion section"""
    f.write("\n## Conclusion\n\n")
    f.write("By grouping Versicolor and Virginica together, we created a binary classification problem ")
    f.write("that allows for clear visualization and interpretation of the SVM's weight vector and decision boundary. ")
    f.write("The linear kernel provides an explicit mathematical representation of how the model separates the classes ")
    f.write("based on the two features (Petal Length and Petal Width).\n")
