# ============================================================
# COMPLETE KNN IMPLEMENTATION — All concepts covered
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_digits, load_breast_cancer

# ============================================================
# PART 1 — KNN from scratch (so you understand what's inside)
# ============================================================

class KNNFromScratch:
    """KNN built from scratch — no sklearn"""

    def __init__(self, k=3, distance='euclidean'):
        self.k = k
        self.distance = distance

    def fit(self, X_train, y_train):
        # No training needed — just store the data
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def _distance(self, a, b):
        if self.distance == 'euclidean':
            return np.sqrt(np.sum((a - b) ** 2))
        elif self.distance == 'manhattan':
            return np.sum(np.abs(a - b))

    def predict_single(self, x):
        # Step 1: compute distance to every training point
        distances = [self._distance(x, xi) for xi in self.X_train]

        # Step 2: sort and get top-k indices
        k_indices = np.argsort(distances)[:self.k]

        # Step 3: get labels of top-k neighbors
        k_labels = self.y_train[k_indices]

        # Step 4: majority vote
        values, counts = np.unique(k_labels, return_counts=True)
        return values[np.argmax(counts)]

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

# Quick test
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn_scratch = KNNFromScratch(k=5)
knn_scratch.fit(X_train, y_train)
print(f"From-scratch KNN accuracy: {knn_scratch.score(X_test, y_test):.3f}")


# ============================================================
# PART 2 — Activity 1: Iris Classification (from lab)
# ============================================================

print("\n===== ACTIVITY 1: IRIS DATASET =====")

from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=iris.target_names))


# ============================================================
# PART 3 — Activity 2: Breast Cancer (from lab)
# ============================================================

print("\n===== ACTIVITY 2: BREAST CANCER =====")

from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()
X = pd.DataFrame(bc.data, columns=bc.feature_names)
X = X[['mean area', 'mean compactness']]      # Two features for easy plotting
y = pd.Categorical.from_codes(bc.target, bc.target_names)
y = pd.get_dummies(y, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"Accuracy: {cm.diagonal().sum() / cm.sum():.1%}")


# ============================================================
# PART 4 — LAB TASK 1: Digits Dataset
# ============================================================

print("\n===== LAB TASK 1: DIGITS DATASET =====")

digits = load_digits()
X_d, y_d = digits.data, digits.target

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_d, y_d, test_size=0.2, random_state=42
)

# --- Method A: Try values manually ---
print("Accuracy for different K values:")
for k in [1, 3, 5, 7, 9, 11, 15]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_d, y_train_d)
    acc = knn.score(X_test_d, y_test_d)
    print(f"  K={k:2d} → accuracy: {acc:.4f}")

# --- Method B: GridSearchCV (finds best K automatically) ---
print("\nUsing GridSearchCV to find optimal K...")
param_grid = {'n_neighbors': list(range(1, 21))}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_d, y_train_d)
print(f"Best K: {grid.best_params_['n_neighbors']}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")

# Evaluate best model
best_knn = grid.best_estimator_
y_pred_d = best_knn.predict(X_test_d)
print(f"Test accuracy: {accuracy_score(y_test_d, y_pred_d):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_d, y_pred_d))

# --- Plot 1: Confusion Matrix ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_d = confusion_matrix(y_test_d, y_pred_d)
disp = ConfusionMatrixDisplay(cm_d, display_labels=digits.target_names)
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f'Confusion Matrix (Best K={grid.best_params_["n_neighbors"]})')

# --- Plot 2: K vs Accuracy ---
cv_results = grid.cv_results_
axes[1].plot(range(1, 21), cv_results['mean_test_score'], 'o-', color='steelblue', linewidth=2)
axes[1].fill_between(range(1, 21),
    cv_results['mean_test_score'] - cv_results['std_test_score'],
    cv_results['mean_test_score'] + cv_results['std_test_score'],
    alpha=0.2, color='steelblue')
axes[1].axvline(grid.best_params_['n_neighbors'], color='coral', linestyle='--', label=f"Best K")
axes[1].set_xlabel('K (number of neighbors)')
axes[1].set_ylabel('CV Accuracy')
axes[1].set_title('K vs Accuracy (with std deviation)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('digits_knn_results.png', dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# PART 5 — LAB TASK 2: Abalone Age Prediction (KNN Regression)
# ============================================================

print("\n===== LAB TASK 2: ABALONE AGE PREDICTION =====")

# Download dataset (or load from CSV after downloading)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight',
           'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
df = pd.read_csv(url, header=None, names=columns)

# Rings + 1.5 = Age (given by dataset docs)
df['Age'] = df['Rings'] + 1.5
print(df.head())
print(f"\nDataset shape: {df.shape}")
print(f"Age range: {df['Age'].min():.1f} to {df['Age'].max():.1f} years")

# --- Encode categorical: Sex (M/F/I) → one-hot ---
df_encoded = pd.get_dummies(df, columns=['Sex'], drop_first=False)

X_ab = df_encoded.drop(['Rings', 'Age'], axis=1)
y_ab = df_encoded['Age']

# --- Feature scaling is CRUCIAL for KNN ---
# KNN uses distances — unscaled features with large ranges dominate!
scaler = StandardScaler()
X_ab_scaled = scaler.fit_transform(X_ab)

X_train_ab, X_test_ab, y_train_ab, y_test_ab = train_test_split(
    X_ab_scaled, y_ab, test_size=0.2, random_state=42
)

# --- Find best K using GridSearchCV ---
param_grid_r = {'n_neighbors': [3, 5, 7, 9, 11, 15, 20]}
grid_r = GridSearchCV(
    KNeighborsRegressor(),
    param_grid_r,
    cv=5,
    scoring='r2',     # R² score for regression
    n_jobs=-1
)
grid_r.fit(X_train_ab, y_train_ab)
print(f"\nBest K for regression: {grid_r.best_params_['n_neighbors']}")
print(f"Best CV R²: {grid_r.best_score_:.4f}")

best_knn_r = grid_r.best_estimator_
y_pred_ab = best_knn_r.predict(X_test_ab)

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
mae  = mean_absolute_error(y_test_ab, y_pred_ab)
rmse = np.sqrt(mean_squared_error(y_test_ab, y_pred_ab))
r2   = r2_score(y_test_ab, y_pred_ab)

print(f"\nTest Results:")
print(f"  MAE  = {mae:.3f} years")
print(f"  RMSE = {rmse:.3f} years")
print(f"  R²   = {r2:.3f}")

# --- Plot: actual vs predicted ---
plt.figure(figsize=(7, 5))
plt.scatter(y_test_ab, y_pred_ab, alpha=0.4, color='steelblue', s=15)
plt.plot([y_ab.min(), y_ab.max()], [y_ab.min(), y_ab.max()], 'r--', linewidth=2)
plt.xlabel('Actual Age (years)')
plt.ylabel('Predicted Age (years)')
plt.title(f'KNN Regression — Abalone Age (K={grid_r.best_params_["n_neighbors"]}, R²={r2:.3f})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('abalone_knn_results.png', dpi=150, bbox_inches='tight')
plt.show()