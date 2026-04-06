# K-Nearest Neighbors (KNN) Implementation Report

This report documents the implementation of the KNN algorithm, compares different distance metrics, and explains the theoretical foundations of the algorithm.

## 1. The KNN Algorithm Steps

The K-Nearest Neighbors algorithm is a non-parametric, lazy learning algorithm used for classification and regression. The steps are as follows:

1.  **Select the value of K**: Decide the number of neighbors to consider (e.g., K=3).
2.  **Calculate distance**: Compute the distance (Euclidean, Manhattan, etc.) between the test data point and all the training data points.
3.  **Find closest neighbors**: Sort the calculated distances and select the K points with the smallest distances.
4.  **Vote on labels**: Count the number of each class label among the K neighbors.
5.  **Assign label**: Assign the class with the highest frequency (majority vote) to the test data point.

---

## 2. Distance Metrics Comparison

In this assignment, we compared two primary distance metrics:

### Euclidean Distance
- **Formula**: $d(x, y) = \sqrt{\sum (x_i - y_i)^2}$
- **Description**: The "straight-line" distance between two points in Euclidean space. It is the most common metric used in KNN.

### Manhattan Distance
- **Formula**: $d(x, y) = \sum |x_i - y_i|$
- **Description**: Also known as "city block" distance. It calculates the distance by summing the absolute differences of their coordinates.

**Comparison Result on Iris Dataset**:
- Both metrics performed exceptionally well on the Iris dataset, achieving nearly identical accuracy. This is common for small, well-separated datasets like Iris.

---

## 3. The Effect of K-Values

The selection of **K** is a critical hyperparameter that influences the model's performance:

- **Small K (e.g., K=1 or K=3)**:
  - **Risk**: Overfitting.
  - **Effect**: The model becomes sensitive to noise in the data. The decision boundary becomes complex and "wiggly."
  - **Result**: High variance, low bias.

- **Large K (e.g., K=20 or K=50)**:
  - **Risk**: Underfitting.
  - **Effect**: The model ignores local patterns and becomes too smooth. It may miss smaller transitions between classes.
  - **Result**: Low variance, high bias.

- **Optimal K**: Typically found using techniques like Cross-Validation. For many small datasets, $K=3$ or $K=5$ provides a good balance between bias and variance.

---

## 4. Why Use KNN?
- **Pros**: Simple to implement, no training phase required, naturally handles multi-class problems.
- **Cons**: High prediction cost (Lazy Learning), sensitive to the scale of features, and struggles with high-dimensional data (the "curse of dimensionality").
