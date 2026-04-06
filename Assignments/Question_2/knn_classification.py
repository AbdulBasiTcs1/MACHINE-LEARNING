# ============================================================
# QUESTION 2: KNN Classification (k=3) — Euclidean vs Manhattan
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset
print("Loading iris.csv dataset...")
df = pd.read_csv("iris.csv")
print(df.head())

# 2. Encode Labels
le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['species'])

# 3. Prepare Features (X) and Target (y)
X = df.drop(['species', 'species_encoded'], axis=1)
y = df['species_encoded']

# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Implement KNN with k=3 and Euclidean Distance
print("\n--- Model A: KNN (k=3, Euclidean) ---")
knn_euclidean = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn_euclidean.fit(X_train, y_train)
y_pred_e = knn_euclidean.predict(X_test)
acc_e = accuracy_score(y_test, y_pred_e)
print(f"Euclidean Accuracy: {acc_e:.4f} ({acc_e*100:.2f}%)")

# 5. Implement KNN with k=3 and Manhattan Distance
print("\n--- Model B: KNN (k=3, Manhattan) ---")
knn_manhattan = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
knn_manhattan.fit(X_train, y_train)
y_pred_m = knn_manhattan.predict(X_test)
acc_m = accuracy_score(y_test, y_pred_m)
print(f"Manhattan Accuracy: {acc_m:.4f} ({acc_m*100:.2f}%)")

# 6. Evaluation Comparison
print("\nComparison Result:")
if acc_e == acc_m:
    print("Both metrics gave identical accuracy on this dataset.")
elif acc_e > acc_m:
    print(f"Euclidean Distance performed better ({acc_e:.4f} vs {acc_m:.4f}).")
else:
    print(f"Manhattan Distance performed better ({acc_m:.4f} vs {acc_e:.4f}).")

# 7. Detailed Report for Euclidean (Standard)
print("\nClassification Report (Euclidean):")
print(classification_report(y_test, y_pred_e, target_names=le.classes_))

# 8. Plot Accuracy Comparison
metrics = ['Euclidean', 'Manhattan']
accuracies = [acc_e, acc_m]

plt.figure(figsize=(7, 5))
sns.barplot(x=metrics, y=accuracies, palette='viridis')
plt.title('KNN (k=3) Accuracy: Euclidean vs Manhattan')
plt.ylim(0.9, 1.05)
plt.ylabel('Accuracy Score')
plt.grid(axis='y', alpha=0.3)
plt.savefig('distance_comparison.png')
print("\nAccuracy comparison plot saved as 'distance_comparison.png'.")
