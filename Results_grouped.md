# Binary Classification with Grouped Classes using Support Vector Machine

## Overview

This analysis groups two of the three Iris classes together to perform binary classification:
- **Class 0**: Setosa
- **Class 1**: Versicolor + Virginica (grouped as 'Non-Setosa')

This grouping allows us to calculate and analyze the **weight vector (w)** and support vectors for a linear decision boundary.

## Dataset Information

- **Dataset**: Iris Dataset (Binary Classification with Grouped Classes)
- **Original Classes**: Setosa, Versicolor, Virginica
- **Grouped Classes**: Setosa vs Non-Setosa (Versicolor + Virginica)
- **Features Used**: Petal Length (cm), Petal Width (cm)
- **Total Samples**: 150
- **Training Samples**: 105
- **Testing Samples**: 45

## SVM Model Configuration

- **Kernel**: Linear (enables calculation of explicit weight vector)
- **C Parameter**: 1.0
- **Feature Scaling**: StandardScaler

## Hyperplane Parameters

### Weight Vector (w)

The weight vector defines the orientation of the decision boundary:

```
w = [0.995574, 1.226633]
```

- **w₁** (Petal Length (cm)): 0.995574
- **w₂** (Petal Width (cm)): 1.226633

### Hyperplane Equation

The decision boundary is defined by the equation:

```
0.995574 × Petal Length (cm) + 1.226633 × Petal Width (cm) + 1.455014 = 0
```

### Geometric Properties

- **Bias (b)**: 1.455014
- **||w|| (Weight vector magnitude)**: 1.579809
- **Margin**: 1.265976 (distance = 2/||w||)

#### Interpretation

- The **weight vector w** is perpendicular to the decision boundary hyperplane
- The direction of w points from the negative class (Setosa) toward the positive class (Non-Setosa)
- The **margin** represents the perpendicular distance between the hyperplane and the nearest support vectors
- A larger margin indicates better separation between classes

## Support Vectors

- **Total Support Vectors**: 4
- **Support Vectors for Setosa**: 2
- **Support Vectors for Non-Setosa (Versicolor + Virginica)**: 2

Support vectors are the critical data points that:
- Lie on or within the margin boundaries
- Directly influence the position and orientation of the decision boundary
- Define the maximum margin hyperplane

## Model Performance

### Accuracy Scores

- **Training Accuracy**: 1.0000 (100.00%)
- **Testing Accuracy**: 1.0000 (100.00%)

### Confusion Matrix

```
                     Predicted Setosa          Predicted Non-Setosa (Versicolor + Virginica)
Actual Setosa          15                               0
Actual Non-Setosa (Versicolor + Virginica) 0                                30
```

### Classification Report

```
                                     precision    recall  f1-score   support

                             Setosa       1.00      1.00      1.00        15
Non-Setosa (Versicolor + Virginica)       1.00      1.00      1.00        30

                           accuracy                           1.00        45
                          macro avg       1.00      1.00      1.00        45
                       weighted avg       1.00      1.00      1.00        45

```

## Visualizations

### Main Visualization

![SVM Grouped Visualization](svm_visualization_grouped.png)

**Interpretation:**
- **Left Plot**: Shows the decision boundary (solid line), margin boundaries (dashed lines), support vectors (green circles), and the weight vector w (purple arrow)
- **Middle Plot**: Original three-class data with support vectors highlighted
- **Right Plot**: Detailed breakdown of hyperplane parameters and weight vector components

### Weight Vector Analysis

![Weight Vector Analysis](weight_vector_analysis.png)

## Mathematical Details

### SVM Decision Function

For a linear SVM, the decision function is:

```
f(x) = w · x + b
     = w₁×x₁ + w₂×x₂ + b
```

Where:
- **w** = weight vector (normal to the hyperplane)
- **x** = feature vector [x₁, x₂]
- **b** = bias term (intercept)

### Classification Rule

```
If f(x) ≥ 0: predict 'Non-Setosa (Versicolor + Virginica)'
If f(x) < 0: predict 'Setosa'
```

### Margin Calculation

The margin between the two support vector boundaries is:

```
Margin = 2 / ||w||
       = 2 / 1.579809
       = 1.265976
```

## Key Findings

- The SVM model achieved **excellent** classification performance.
- The decision boundary is defined by **4 support vectors** (3.8% of training data).
- Linear kernel is sufficient to separate Setosa from Non-Setosa classes.
- The weight vector w = [0.9956, 1.2266] indicates that **Petal Width (cm)** has a stronger influence on the classification decision.
- The model generalizes well to unseen data.

## Conclusion

By grouping Versicolor and Virginica together, we created a binary classification problem that allows for clear visualization and interpretation of the SVM's weight vector and decision boundary. The linear kernel provides an explicit mathematical representation of how the model separates the classes based on the two features (Petal Length and Petal Width).
