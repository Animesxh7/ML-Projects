# PolyPredict - Polynomial Regression

A simple machine learning project to understand and experiment with Polynomial Regression using the Wine Quality dataset.

## Overview

PolyPredict explores how Polynomial Regression can capture non-linear relationships between input features and a target variable.

The project compares ordinary Linear Regression with Polynomial Regression and provides an interactive Streamlit application to experiment with model predictions.

## Objectives

- Understand the working of Polynomial Regression.
- Explore polynomial feature transformation.
- Compare Linear and Polynomial Regression.
- Evaluate models using regression metrics.
- Visualize actual vs predicted values.
- Experiment with different polynomial degrees.

## Dataset

**Wine Quality - White Wine Dataset**

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/186/wine+quality)

- Original records: 4,898
- Input features: 11
- Target variable: `quality`
- File: `winequality-white.csv`

For this project, three features are selected:

| Feature          | Description            |
| ---------------- | ---------------------- |
| alcohol          | Alcohol content        |
| volatile acidity | Volatile acidity level |
| sulphates        | Sulphates content      |

**Target:** `quality`

The dataset is manually downloaded and loaded using Pandas.

## Project Structure

```text
PolyPredict/
│
├──winequality-white.csv
├── app.py
├── PolyPredict_Polynomial_Regression.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook

## Machine Learning Workflow

1. Load the dataset using Pandas.
2. Perform data inspection and exploratory data analysis.
3. Check missing values and duplicate records.
4. Select relevant input features.
5. Split the data into training and testing sets.
6. Train a Linear Regression baseline.
7. Generate polynomial features.
8. Train the Polynomial Regression model.
9. Compare model performance.
10. Visualize predictions and residuals.

## Models

### 1. Linear Regression

Used as the baseline model to understand the relationship between the selected features and wine quality.

### 2. Polynomial Regression

Uses `PolynomialFeatures` to transform the original input features into polynomial combinations before fitting Linear Regression.

The application allows experimentation with polynomial degrees 2, 3 and 4.

## Evaluation Metrics

| Metric   | Purpose                                                           |
| -------- | ----------------------------------------------------------------- |
| MAE      | Measures average absolute prediction error                        |
| RMSE     | Measures prediction error while penalizing larger errors          |
| R² Score | Measures the proportion of target variance explained by the model |

## Streamlit Application

The application includes:

- Dataset overview and summary statistics.
- Polynomial degree selection.
- Training and testing split configuration.
- Linear vs Polynomial Regression comparison.
- Actual vs predicted visualization.
- Residual distribution.
- Interactive wine quality prediction.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Animesxh7/PolyPredict.git
cd PolyPredict
```

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download `winequality-white.csv` from the UCI repository and place it in the expected dataset location.

### 5. Run the Jupyter Notebook

```bash
jupyter notebook
```

Open `PolyPredict_Polynomial_Regression.ipynb` and execute the cells.

### 6. Run the Streamlit application

```bash
python -m streamlit run app.py
```

## Key Learning

Polynomial Regression is still a linear model with respect to its coefficients. It captures more complex relationships by transforming input features into polynomial terms.

This project demonstrates how feature engineering can improve a regression model's ability to represent non-linear patterns.

---

*Built as part of a hands-on Machine Learning concept demonstration series.*
