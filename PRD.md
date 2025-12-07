# Product Requirements Document (PRD)
# SVM Classification Analysis on Iris Dataset

## 1. Project Overview

### 1.1 Purpose
Create a comprehensive educational project demonstrating Support Vector Machine (SVM) classification using the Iris dataset. The project showcases binary and multi-class classification scenarios with detailed visualizations and mathematical analysis.

### 1.2 Objectives
- Implement SVM classifiers for different classification scenarios
- Provide clear visualizations of decision boundaries and support vectors
- Calculate and explain the weight vector (w) for linear SVMs
- Analyze class separability and centroid distances
- Generate comprehensive reports with performance metrics

### 1.3 Target Audience
- Machine learning students and practitioners
- Data science educators
- Researchers studying classification algorithms
- Anyone learning about Support Vector Machines

## 2. Functional Requirements

### 2.1 Classification Scenarios

#### 2.1.1 Binary Classification (2 Classes)
- **Script**: `svm_classification.py`
- **Classes**: Setosa vs Versicolor
- **Features**: Sepal Length, Sepal Width
- **Kernel**: Linear
- **Deliverables**:
  - Training and testing accuracy
  - Confusion matrix
  - Classification report
  - Visualization showing decision boundary and support vectors
  - Results output: `Results.md`

#### 2.1.2 Multi-class Classification (3 Classes)
<!-- - **Script**: `svm_classification_3classes.py`
- **Classes**: Setosa, Versicolor, Virginica
- **Features**: Petal Length, Petal Width
- **Kernel**: RBF (Radial Basis Function)
- **Strategy**: One-vs-Rest (OvR)
- **Deliverables**:
  - Training and testing accuracy
  - Confusion matrix and heatmap
  - Classification report
  - Three-panel visualization (decision boundaries, original data, support vector distribution)
  - Results output: `Results_3_classes.md` -->

#### 2.1.3 Grouped Binary Classification (Weight Vector Analysis)
- **Script**: `svm_classification_grouped.py`
- **Classes**: Setosa vs Non-Setosa (Versicolor + Virginica grouped)
- **Features**: Petal Length, Petal Width
- **Kernel**: Linear
- **Deliverables**:
  - Explicit weight vector (w) calculation
  - Hyperplane equation
  - Margin calculation
  - Bias term (b)
  - Three-panel visualization with weight vector
  - Weight vector component analysis
  - Results output: `Results_grouped.md`

### 2.2 Analysis Tools

#### 2.2.1 Class Separation Analysis
- **Script**: `analyze_class_separation.py`
- **Purpose**: Explain why Versicolor and Virginica were grouped
- **Deliverables**:
  - Statistical properties per class
  - Inter-class distance analysis
  - Feature-wise overlap analysis
  - 9-panel comprehensive visualization
  - Output: `class_separation_analysis.png`

#### 2.2.2 Centroid Visualization
- **Script**: `plot_centroids.py`
- **Purpose**: Visualize class centroids on XY plane
- **Deliverables**:
  - Centroid coordinates
  - Pairwise distance calculations
  - Three-panel visualization
  - Simplified single-panel view
  - Outputs: `centroids_visualization.png`, `centroids_simple.png`

### 2.3 Documentation

#### 2.3.1 Grouping Explanation
- **File**: `GROUPING_EXPLANATION.md`
- **Content**:
  - Statistical evidence for grouping decision
  - Centroid distance analysis with embedded graphs
  - Feature-wise overlap analysis
  - Mathematical impact
  - Visual evidence with multiple graphs

#### 2.3.2 Results Reports
- **Files**: `Results.md`, `Results_3_classes.md`, `Results_grouped.md`
- **Content**:
  - Dataset information
  - Model configuration
  - Support vector details
  - Performance metrics
  - Visualizations
  - Key findings and interpretation

## 3. Technical Requirements

### 3.1 Dependencies
- Python 3.7+
- NumPy >= 1.21.0
- Scikit-learn >= 1.0.0
- Matplotlib >= 3.5.0
- SciPy >= 1.7.0

### 3.2 Data Requirements
- Iris dataset (150 samples, 4 features, 3 classes)
- Train/test split: 70/30
- Stratified sampling to maintain class balance

### 3.3 Performance Requirements
- All scripts should execute in under 10 seconds
- Visualizations should be saved at 300 DPI
- Accuracy targets:
  - Binary classification (2 classes): > 95%
  - Multi-class (3 classes): > 85%
  - Grouped binary: > 95%

### 3.4 Output Requirements
- All visualizations saved as PNG files
- All results saved as Markdown files
- Console output showing progress and key metrics

## 4. Non-Functional Requirements

### 4.1 Code Quality
- Well-documented code with docstrings
- Clear variable names
- Modular functions
- Error handling for edge cases

### 4.2 Usability
- Scripts can be run independently
- Clear console output
- Informative visualizations with legends and labels
- Comprehensive markdown reports

### 4.3 Reproducibility
- Fixed random seeds (random_state=42)
- Deterministic results
- Version-controlled dependencies

### 4.4 Educational Value
- Clear explanations in output
- Visual representations of concepts
- Mathematical formulas and equations
- Interpretation of results

## 5. Success Criteria

### 5.1 Functionality
- ✓ All scripts execute without errors
- ✓ All visualizations generated correctly
- ✓ All markdown reports created with proper formatting
- ✓ Accuracy targets met or exceeded

### 5.2 Documentation
- ✓ Comprehensive README with usage instructions
- ✓ Clear explanations of grouping decisions
- ✓ All visualizations properly labeled
- ✓ Mathematical concepts clearly explained

### 5.3 Educational Impact
- ✓ Demonstrates SVM concepts effectively
- ✓ Shows different kernel types
- ✓ Explains support vectors and weight vectors
- ✓ Provides insights into class separability

## 6. Project Deliverables

### 6.1 Python Scripts
1. `svm_classification.py` - Binary classification (2 classes)
2. `svm_classification_3classes.py` - Multi-class classification
3. `svm_classification_grouped.py` - Grouped binary with weight vector
4. `analyze_class_separation.py` - Class separation analysis
5. `plot_centroids.py` - Centroid visualization

### 6.2 Documentation Files
1. `README.md` - Project overview and usage guide
2. `PRD.md` - Product requirements (this document)
3. `TASKS.md` - Task tracking and project status
4. `GROUPING_EXPLANATION.md` - Explanation of grouping decision
5. `requirements.txt` - Python dependencies
6. `.gitignore` - Git ignore patterns

### 6.3 Results Files
1. `Results.md` - Binary classification results
2. `Results_3_classes.md` - Multi-class results
3. `Results_grouped.md` - Grouped classification results

### 6.4 Visualizations
1. `svm_visualization.png` - Binary classification visualization
2. `svm_visualization_3classes.png` - Multi-class visualization
3. `confusion_matrix_3classes.png` - Confusion matrix heatmap
4. `svm_visualization_grouped.png` - Grouped classification visualization
5. `weight_vector_analysis.png` - Weight vector breakdown
6. `class_separation_analysis.png` - Class separation analysis
7. `centroids_visualization.png` - Centroid three-panel view
8. `centroids_simple.png` - Centroid simplified view

## 7. Future Enhancements

### 7.1 Potential Additions
- Interactive visualizations using Plotly
- Hyperparameter tuning analysis
- Comparison with other classifiers (Random Forest, Neural Networks)
- Cross-validation analysis
- ROC curves and AUC scores
- Feature importance analysis
- 3D visualizations using all 4 features

### 7.2 Extended Analysis
- Grid search for optimal parameters
- Kernel comparison (Linear vs RBF vs Polynomial)
- Sensitivity analysis
- Learning curves
- Support vector sensitivity analysis

## 8. Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0 | 2025-12-07 | Initial release with all core functionality |

## 9. Stakeholders

- **Project Owner**: Educational/Research Team
- **Primary Users**: ML Students, Data Science Learners
- **Contributors**: Development Team
- **Reviewers**: ML Educators, Domain Experts

## 10. Constraints and Assumptions

### 10.1 Constraints
- Limited to Iris dataset
- 2D visualizations for clarity
- Python-only implementation
- No GUI interface

### 10.2 Assumptions
- Users have Python 3.7+ installed
- Users have basic understanding of machine learning
- Users can read markdown files
- Users have necessary Python packages installed

---

**Document Status**: Final
**Last Updated**: 2025-12-07
**Approved By**: Project Team
