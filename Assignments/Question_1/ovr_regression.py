# ============================================================
# QUESTION 1: Multiclass Linear Regression using One-vs-Rest (OvR)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load Dataset
print("Loading iris.csv dataset...")
df = pd.read_csv("iris.csv")
print(df.head())

# 2. Encode Labels
# Species: Iris-setosa, Iris-versicolor, Iris-virginica -> 0, 1, 2
print("\nEncoding species labels...")
le = LabelEncoder()
df['species_encoded'] = le.fit_transform(df['species'])
print(f"Classes found: {list(le.classes_)}")
print(f"Encoded mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 3. Prepare Features (X) and Target (y)
X = df.drop(['species', 'species_encoded'], axis=1)
y = df['species_encoded']

# Split into Training (80%) and Testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model (One-vs-Rest strategy)
# Note: LogisticRegression is the standard "Linear Model" for classification tasks.
print("\nTraining Multiclass Logistic Regression (One-vs-Rest)...")
model = LogisticRegression(multi_class='ovr', solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)

# 5. Predict Output
y_pred = model.predict(X_test)

# 6. Evaluate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 7. Visualization: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2f})')
plt.tight_layout()
plt.savefig('confusion_matrix_ovr.png')
print("\nConfusion matrix plot saved as 'confusion_matrix_ovr.png'.")

# Manual Prediction Example
sample = [[5.1, 3.5, 1.4, 0.2]] # Typical Setosa
pred_class = model.predict(sample)
pred_species = le.inverse_transform(pred_class)
print(f"\nSample Prediction for {sample}: {pred_species[0]}")
