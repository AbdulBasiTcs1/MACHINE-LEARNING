# --- Q2: KNN Classification ---


import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# CUSTOM KNN IMPLEMENTATION FROM SCRATCH
class KNNClassifier:
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        return self

    def _distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _predict_single(self, x):
        distances = [self._distance(x, x_tr) for x_tr in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0]
        return most_common

    def predict(self, X_test):
        return np.array([self._predict_single(x) for x in np.array(X_test)])

    def score(self, X_test, y_test):
        return accuracy_score(y_test, self.predict(X_test))


# 1. LOAD DATASET
df = pd.read_csv("wine.csv")
feature_cols = [c for c in df.columns if c not in ['target', 'target_name']]
X = df[feature_cols].values
y = df['target'].values

# 2. TRAIN / TEST SPLIT + SCALING
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# 3. KNN WITH k=3, EUCLIDEAN DISTANCE
knn_euclidean = KNNClassifier(k=3, metric='euclidean')
knn_euclidean.fit(X_train_sc, y_train)
y_pred_euc = knn_euclidean.predict(X_test_sc)
acc_euc = accuracy_score(y_test, y_pred_euc)

print(f"KNN (k=3, Euclidean Distance) Accuracy: {acc_euc * 100:.2f}%")
print("\nClassification Report (Euclidean):")
print(classification_report(y_test, y_pred_euc,
      target_names=['class_0', 'class_1', 'class_2']))

# 4. KNN WITH k=3, MANHATTAN DISTANCE
knn_manhattan = KNNClassifier(k=3, metric='manhattan')
knn_manhattan.fit(X_train_sc, y_train)
y_pred_man = knn_manhattan.predict(X_test_sc)
acc_man = accuracy_score(y_test, y_pred_man)

print(f"\nKNN (k=3, Manhattan Distance) Accuracy: {acc_man * 100:.2f}%")
print("\nClassification Report (Manhattan):")
print(classification_report(y_test, y_pred_man,
      target_names=['class_0', 'class_1', 'class_2']))

# 5. Compare Euclidean vs Manhattan
print("\nDistance Metric Comparison:")
print(f"  Euclidean Accuracy: {acc_euc*100:.2f}%")
print(f"  Manhattan Accuracy: {acc_man*100:.2f}%")
