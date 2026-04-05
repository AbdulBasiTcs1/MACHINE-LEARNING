# Machine learning algorithms (01–03)

Hands-on Python and browser demos for core ML topics. This README covers the numbered algorithm folders **01**, **02**, and **03** only.

## Prerequisites

- **Python 3** with: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`
- **Network access** (optional): `02_K-NEAREST-NEIGHBOUR/KnnCode.py` loads the Abalone dataset from the UCI ML Repository

---

## 01 — Decision trees (`01_Decision-Trees`)

Placeholder folder for **decision tree** material. There is no code here yet; content can be added when you build that module.

---

## 02 — K-nearest neighbours (`02_K-NEAREST-NEIGHBOUR`)

| File | Description |
|------|-------------|
| `KnnCode.py` | End-to-end KNN: implementation from scratch, then scikit-learn for classification and regression |
| `knn_algorithm_interactive.html` | Interactive canvas: click to classify a point; adjust **K** and see majority vote among three classes |

**`KnnCode.py` outline**

- **From scratch:** `KNNFromScratch` (Euclidean / Manhattan), tested on Iris
- **Classification:** Iris, Breast Cancer (two features), Digits (tuning **K**, `GridSearchCV`, confusion matrix and accuracy plot)
- **Regression:** Abalone age via `KNeighborsRegressor` (data from URL), scaling, `GridSearchCV`, MAE / RMSE / R²

**Run the script**

```bash
cd 02_K-NEAREST-NEIGHBOUR
python KnnCode.py
```

Plots may be saved as `digits_knn_results.png` and `abalone_knn_results.png` in the same folder.

**Interactive HTML**

Open `knn_algorithm_interactive.html` in a browser (double-click or drag into a tab). No build step.

---

## 03 — Linear & related regression (`03_LINEAR-REGRESSION`)

| File | Description |
|------|-------------|
| `linearRegression.py` | Labs: simple LR, multiple LR, logistic regression, polynomial regression, and extra tasks |
| `lr_residuals_interactive.html` | Interactive residual demo: drag slope/intercept, see MSE and “best fit” |

**`linearRegression.py` topics**

- Simple linear regression — `Salary_Data.csv` (experience vs salary), metrics, plots
- Multiple linear regression — `50_Startups.csv` (with one-hot state encoding)
- Logistic regression — `insurance_data.csv` (binary), then multiclass on Digits
- Lab-style tasks — `canada_per_capita_income.csv`, `hiring.csv`, `Position_Salaries.csv` (polynomial pipeline)

**Run the script**

```bash
cd 03_LINEAR-REGRESSION
python linearRegression.py
```

Place the CSV files **in the same directory** as `linearRegression.py` so `pd.read_csv(...)` resolves. The script saves figures such as `activity1_salary.png`, `task5_polynomial.png`, etc., when those sections run.

**Interactive HTML**

Open `lr_residuals_interactive.html` in a browser to explore the regression line, residuals, and total MSE.

---

## Broader repo roadmap

Topics planned for the rest of this repository include: decision trees, K-means, additional classifiers, and Naive Bayes—beyond the 01–03 set described above.
