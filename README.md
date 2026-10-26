# House Price Prediction — Linear Regression

Building a Linear Regression model to predict house prices from features such as area, bedrooms, bathrooms, age, location score, garage type, and condition. The model performed poorly on unseen data, and this project documents why.

## Overview

The project follows a full machine learning workflow: inspect the data, clean it, explore it, prepare features, train a model, test it on unseen houses, and evaluate the results honestly. The most useful part of the result is the diagnosis of why the model failed, which points to the next steps.

**Tools:** Python, Google Colab
**Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
**Dataset:** `homeprices.csv`, 100 houses and 8 columns (area, bedrooms, bathrooms, age, location_score, garage_type, condition, price)

## Process

1. **Data inspection** — Found that `price` was stored as text (one value written as "$550,000"), that `bathrooms` (5), `age` (8), and `garage_type` (8) had missing values, and that `garage_type` had inconsistent capitalization ("detached" vs "Detached"). No duplicate rows.
2. **Data cleaning** — Stripped `$` and `,` from `price` and converted it to numeric. Filled missing numeric values with the median, which is not affected by extreme values, and missing text values with the mode. Standardized capitalization in `garage_type`.
3. **Exploratory data analysis** — Boxplots, histograms, and a correlation heatmap.
4. **Feature preparation** — Target: `price`. Converted `garage_type` and `condition` into 0/1 dummy columns with `get_dummies(drop_first=True)`.
5. **Train/test split** — 80% train (80 houses) and 20% test (20 houses), `random_state=42` for reproducibility.
6. **Training and testing** — Fitted `LinearRegression` on the training set and predicted prices for the unseen test houses.
7. **Evaluation** — RMSE, R², and a residual plot.

## Results

| Metric | Value |
|---|---|
| RMSE | ≈ 160,220 |
| R² | ≈ -0.085 |

- **Strongest driver of price:** `area` (correlation ≈ 0.79), followed by `bathrooms` (0.39) and `location_score` (0.30). `age` is weakly negative (-0.19) and `bedrooms` is weak (0.14).
- **Poor model fit:** Predictions are off by about $160k on average, large next to typical prices of $300k to $800k. The negative R² means the model did worse than predicting the average price for every house.
- **Residual plot:** The points slope downward instead of forming a random cloud around zero. Cheap houses are underpredicted and expensive ones overpredicted, so the constant-error-spread assumption is not satisfied.

## What I learned and next steps

The EDA flagged extreme records, for example a house with 20 bedrooms, a 12,000 sq ft house priced at $4.5M, and one priced at only $45,000. Linear Regression minimizes squared error, so a few extreme rows can pull the fitted line and distort predictions. With more time, I would:

- Remove or correct outliers (for example with the IQR rule), then retrain and compare results
- Test whether a different model handles the data better

## Links

- [Full notebook (Google Colab)](https://colab.research.google.com/drive/1gF7_a75-5o1Wj4E00SSyvyQteqnZN2gD?usp=sharing)
