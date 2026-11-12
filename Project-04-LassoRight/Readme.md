# Project 04 — LassoRight

A small Streamlit concept-demo for **Lasso Regression (L1 regularization)** using the UCI Concrete Compressive Strength dataset.

## What this project demonstrates

- Lasso regression and its L1 penalty
- How the `alpha` parameter controls regularization strength
- Feature scaling before regularized regression
- Coefficient shrinkage and basic feature selection
- Regression evaluation with MAE, RMSE and R²

## Dataset

**Concrete Compressive Strength** from the UCI Machine Learning Repository (dataset ID 165).

- 1,030 rows
- 8 numerical input features
- Target: concrete compressive strength (`strength`, MPa)

The dataset is downloaded locally by `download_data.py`. The CSV is excluded from Git with `.gitignore`; it is not committed to the repository.

## Folder contents

```text
Project-04-LassoRight/
├── app.py
├── LassoRight_Notebook.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── download_data.py
```

After downloading the data, `data.csv` will also appear locally.

## Setup

Python 3.10 or newer is recommended.

### 1. Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download the dataset

```bash
python download_data.py
```

This creates `data.csv` in the project folder.

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

### 5. Open the notebook

```bash
jupyter notebook LassoRight_Notebook.ipynb
```

## App features

- Adjustable Lasso `alpha` and test-set size
- Interactive concrete-mix inputs for prediction
- Actual-versus-predicted scatter plot
- Lasso coefficient chart
- MAE, RMSE and R² test metrics
- Small dataset preview

## Notes

- The app uses a reproducible train/test split controlled by the random-state setting.
- `StandardScaler` is fitted only on the training data through a scikit-learn pipeline.
- The coefficient chart shows coefficients on the standardized feature scale.
- The sample prediction is educational and must not be used for structural engineering decisions.

## Reference

UCI Machine Learning Repository — Concrete Compressive Strength:
https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
