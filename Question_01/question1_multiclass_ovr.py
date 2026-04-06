# --- Q1: Multiclass Linear Regression using One-vs-Rest (OvR) ---


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. LOAD DATASET
df = pd.read_csv("iris.csv")

# 2. ENCODE LABELS
le = LabelEncoder()
X = df[['sepal length (cm)', 'sepal width (cm)',
        'petal length (cm)', 'petal width (cm)']].values
y_raw = df['target_name'].values
y = le.fit_transform(y_raw)

# 3. TRAIN / TEST SPLIT + FEATURE SCALING
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# 4. TRAIN MODEL — One-vs-Rest
base_clf = LogisticRegression(max_iter=200, random_state=42)
ovr_model = OneVsRestClassifier(base_clf)
ovr_model.fit(X_train_sc, y_train)

# 5. PREDICT OUTPUT
y_pred = ovr_model.predict(X_test_sc)

# 6. EVALUATE ACCURACY
acc = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
