# ============================================================
# LAB 03 — REGRESSION  |  Complete Code for All Lab Tasks
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_squared_error,
                              confusion_matrix, classification_report,
                              accuracy_score)
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# ACTIVITY 1 — Simple Linear Regression (Salary vs Experience)
# ============================================================

print("=" * 55)
print("ACTIVITY 1: Simple Linear Regression")
print("=" * 55)

dataset = pd.read_csv("Salary_Data.csv")
print(dataset.head())
print(f"Shape: {dataset.shape}")

X = dataset.iloc[:, :-1].values   # Years of Experience (2D)
y = dataset.iloc[:, -1].values    # Salary (1D)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=0
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing  samples: {len(X_test)}")

# --- Train ---
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# --- Key outputs ---
print(f"\nSlope (b)     = {regressor.coef_[0]:.4f}")
print(f"Intercept (a) = {regressor.intercept_:.4f}")
print(f"Formula: Salary = {regressor.coef_[0]:.2f} × Experience + {regressor.intercept_:.2f}")

# --- Predict ---
y_pred = regressor.predict(X_test)

# --- Evaluation ---
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nR²:   {r2:.4f}  ({r2*100:.1f}% variance explained)")
print(f"RMSE: {rmse:.2f}")

# --- Predict a specific value (from lab: 12 years, 15 years) ---
pred_12 = regressor.predict([[12]])
pred_15 = regressor.predict([[15]])
print(f"\nPredicted salary for 12 years experience: ${pred_12[0]:,.2f}")
print(f"Predicted salary for 15 years experience: ${pred_15[0]:,.2f}")

# Manual check (from lab manual)
manual = 26816.192244031183 + 9345.94244312 * 15
print(f"Manual prediction (15 yrs):               ${manual:,.2f}")

# --- 3 Plots from lab ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(X_train, y_train, color='red', s=30)
axes[0].plot(X_train, regressor.predict(X_train), color='blue', linewidth=2)
axes[0].set_title('Training Set')
axes[0].set_xlabel('Years of Experience')
axes[0].set_ylabel('Salary')

axes[1].scatter(X_test, y_test, color='red', s=30)
axes[1].plot(X_train, regressor.predict(X_train), color='blue', linewidth=2)
axes[1].set_title('Test Set (training line)')
axes[1].set_xlabel('Years of Experience')

axes[2].scatter(X_test, y_test, color='red', s=30)
axes[2].plot(X_test, y_pred, color='blue', linewidth=2)
axes[2].set_title('Test Set (test line)')
axes[2].set_xlabel('Years of Experience')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('activity1_salary.png', dpi=150)
plt.show()


# ============================================================
# ACTIVITY 2 — Multiple Linear Regression (50 Startups)
# ============================================================

print("\n" + "=" * 55)
print("ACTIVITY 2: Multiple Linear Regression")
print("=" * 55)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

dataset2 = pd.read_csv("50_Startups.csv")
print(dataset2.head())

X2 = dataset2.iloc[:, :-1].values
y2 = dataset2.iloc[:, -1].values

# Encode categorical 'State' column (column index 3)
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])],
    remainder='passthrough'
)
X2 = np.array(ct.fit_transform(X2))

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=0
)

regressor2 = LinearRegression()
regressor2.fit(X2_train, y2_train)

y2_pred = regressor2.predict(X2_test)

# Side-by-side: predicted vs actual
comparison = np.concatenate(
    (y2_pred.reshape(-1, 1), y2_test.reshape(-1, 1)), axis=1
)
print("\nPredicted  vs  Actual:")
print(comparison)

r2_mlr = r2_score(y2_test, y2_pred)
rmse_mlr = np.sqrt(mean_squared_error(y2_test, y2_pred))
print(f"\nR²:   {r2_mlr:.4f}")
print(f"RMSE: {rmse_mlr:.2f}")

plt.figure(figsize=(7, 5))
plt.scatter(y2_test, y2_pred, color='steelblue', s=50, edgecolors='white', linewidths=0.8)
plt.plot([y2_test.min(), y2_test.max()],
         [y2_test.min(), y2_test.max()], 'r--', linewidth=1.5)
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.title(f'Multiple LR — Actual vs Predicted  (R²={r2_mlr:.3f})')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('activity2_startups.png', dpi=150)
plt.show()


# ============================================================
# ACTIVITY 3 — Logistic Regression (Binary: Insurance)
# ============================================================

print("\n" + "=" * 55)
print("ACTIVITY 3: Logistic Regression — Binary")
print("=" * 55)

df3 = pd.read_csv("insurance_data.csv")
print(df3.head())

plt.figure(figsize=(7, 4))
plt.scatter(df3.age, df3.bought_insurance, marker='+', color='red', s=60)
plt.xlabel('Age')
plt.ylabel('Bought Insurance (0=No, 1=Yes)')
plt.title('Age vs Insurance Purchase')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('activity3_insurance_scatter.png', dpi=150)
plt.show()

X3 = df3[['age']]
y3 = df3['bought_insurance']
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, train_size=0.8, random_state=42)

log_model = LogisticRegression()
log_model.fit(X3_train, y3_train)

print(f"\nCoefficient (m): {log_model.coef_[0][0]:.4f}")
print(f"Intercept   (b): {log_model.intercept_[0]:.4f}")
print(f"Accuracy: {log_model.score(X3_test, y3_test):.4f}")

# Predict probabilities
proba = log_model.predict_proba(X3_test)
print("\nPrediction probabilities (No | Yes):")
print(proba)

# Sigmoid function (from lab manual)
import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def predict_insurance(age):
    m = log_model.coef_[0][0]
    b = log_model.intercept_[0]
    z = m * age + b
    prob = sigmoid(z)
    decision = "WILL buy" if prob >= 0.5 else "will NOT buy"
    print(f"Age {age}: probability = {prob:.3f} → person {decision} insurance")

predict_insurance(35)   # From lab: 0.485 → no
predict_insurance(43)   # From lab: 0.568 → yes
predict_insurance(25)
predict_insurance(55)


# ============================================================
# ACTIVITY 4 — Logistic Regression Multiclass (Digits)
# ============================================================

print("\n" + "=" * 55)
print("ACTIVITY 4: Logistic Regression — Multiclass (Digits)")
print("=" * 55)

from sklearn.datasets import load_digits

digits = load_digits()

# Visualize first 5 digits
fig, axes = plt.subplots(1, 5, figsize=(10, 2.5))
for i in range(5):
    axes[i].matshow(digits.images[i], cmap='gray')
    axes[i].set_title(f'Label: {digits.target[i]}')
    axes[i].axis('off')
plt.suptitle('Sample digits from dataset')
plt.tight_layout()
plt.savefig('activity4_digits_samples.png', dpi=150)
plt.show()

X4_train, X4_test, y4_train, y4_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

log_multi = LogisticRegression(max_iter=1000)
log_multi.fit(X4_train, y4_train)

acc = log_multi.score(X4_test, y4_test)
print(f"Accuracy: {acc:.4f}  ({acc*100:.1f}%)")

y4_pred = log_multi.predict(X4_test)
print("\nFirst 5 predictions vs actual:")
print(f"Predicted: {log_multi.predict(digits.data[0:5])}")
print(f"Actual:    {digits.target[0:5]}")

# Confusion matrix heatmap
cm4 = confusion_matrix(y4_test, y4_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm4, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.title(f'Confusion Matrix — Digits (Accuracy={acc:.3f})')
plt.tight_layout()
plt.savefig('activity4_digits_cm.png', dpi=150)
plt.show()


# ============================================================
# LAB TASK 1 — Predict Canada's Per Capita Income (2020)
# ============================================================

print("\n" + "=" * 55)
print("LAB TASK 1: Canada Per Capita Income — Predict 2020")
print("=" * 55)

canada = pd.read_csv("canada_per_capita_income.csv")
print(canada.head())
print(f"\nColumns: {list(canada.columns)}")

# Adjust column names to match your CSV file
X_ca = canada.iloc[:, 0].values.reshape(-1, 1)  # year
y_ca = canada.iloc[:, 1].values                  # per capita income

lr_canada = LinearRegression()
lr_canada.fit(X_ca, y_ca)

pred_2020 = lr_canada.predict([[2020]])
print(f"\nPredicted per capita income for 2020: ${pred_2020[0]:,.2f}")
print(f"Slope (b):     {lr_canada.coef_[0]:.4f}")
print(f"Intercept (a): {lr_canada.intercept_:.2f}")

plt.figure(figsize=(8, 5))
plt.scatter(X_ca, y_ca, color='blue', s=25, label='Historical data')
plt.plot(X_ca, lr_canada.predict(X_ca), color='red', linewidth=2, label='Regression line')
plt.scatter([[2020]], pred_2020, color='green', s=100, zorder=5,
            marker='*', label=f'2020 prediction: ${pred_2020[0]:,.0f}')
plt.xlabel('Year')
plt.ylabel('Per Capita Income ($)')
plt.title("Canada's Per Capita Income — Linear Regression Forecast")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task1_canada.png', dpi=150)
plt.show()


# ============================================================
# LAB TASK 2 — Multiple LR: Hiring (Salary Prediction)
# ============================================================

print("\n" + "=" * 55)
print("LAB TASK 2: Hiring — Predict Salary from 3 Features")
print("=" * 55)

hiring = pd.read_csv("hiring.csv")
print(hiring.head())

# Handle missing values in 'experience' column
hiring['experience'] = hiring['experience'].fillna('zero')
experience_map = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,
                  'six':6,'seven':7,'eight':8,'nine':9,'ten':10,
                  'eleven':11,'twelve':12}
hiring['experience'] = hiring['experience'].str.lower().map(experience_map).fillna(0)

# Fill missing test/interview scores with column median
hiring['test_score(out of 10)'] = hiring['test_score(out of 10)'].fillna(
    hiring['test_score(out of 10)'].median()
)

X_hr = hiring[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']].values
y_hr = hiring['salary($)'].values

lr_hire = LinearRegression()
lr_hire.fit(X_hr, y_hr)

# Predict the two candidates from the lab task
candidates = np.array([
    [2, 9, 6],    # 2 yr exp, 9 test, 6 interview
    [12, 10, 10]  # 12 yr exp, 10 test, 10 interview
])
predicted_salaries = lr_hire.predict(candidates)

print(f"\nCandidate 1 (2yr exp, test:9, interview:6): ${predicted_salaries[0]:,.2f}")
print(f"Candidate 2 (12yr exp, test:10, interview:10): ${predicted_salaries[1]:,.2f}")
print(f"\nCoefficients: {lr_hire.coef_}")
print(f"Intercept: {lr_hire.intercept_:.2f}")


# ============================================================
# LAB TASK 5 — Polynomial Regression (Position Salaries)
# ============================================================

print("\n" + "=" * 55)
print("LAB TASK 5: Polynomial Regression")
print("=" * 55)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

pos_sal = pd.read_csv("Position_Salaries.csv")
print(pos_sal.head())

X_ps = pos_sal.iloc[:, 1:-1].values   # Level
y_ps = pos_sal.iloc[:, -1].values     # Salary

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
degrees = [1, 2, 4]

for i, deg in enumerate(degrees):
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=deg)),
        ('lr',   LinearRegression())
    ])
    poly_pipeline.fit(X_ps, y_ps)

    X_plot = np.linspace(X_ps.min(), X_ps.max(), 200).reshape(-1, 1)
    y_plot = poly_pipeline.predict(X_plot)
    r2_p = r2_score(y_ps, poly_pipeline.predict(X_ps))

    axes[i].scatter(X_ps, y_ps, color='red', s=50, zorder=5)
    axes[i].plot(X_plot, y_plot, color='blue', linewidth=2)
    axes[i].set_title(f'Degree {deg}  (R²={r2_p:.3f})')
    axes[i].set_xlabel('Position Level')
    axes[i].set_ylabel('Salary')
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Polynomial Regression — Increasing degree fits curves better', y=1.02)
plt.tight_layout()
plt.savefig('task5_polynomial.png', dpi=150)
plt.show()

# Predict salary for level 6.5 (between Manager and Director)
best_poly = Pipeline([
    ('poly', PolynomialFeatures(degree=4)),
    ('lr',   LinearRegression())
])
best_poly.fit(X_ps, y_ps)
pred_65 = best_poly.predict([[6.5]])
print(f"\nPredicted salary for position level 6.5: ${pred_65[0]:,.2f}")