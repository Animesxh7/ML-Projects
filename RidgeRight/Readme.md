# RidgeRight — Ridge Regression

A small interactive project demonstrating **Ridge Regression (L2 regularization)** by comparing it with ordinary Linear Regression.

## Dataset
Download the [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance). Extract the CSV, rename it to **`insurance.csv`**, and place it beside `app.py`.

The project loads the file using:

```python
df = pd.read_csv("insurance.csv")
```

The target column is `charges`. The dataset contains age, sex, BMI, children, smoker, region and insurance charges. The Kaggle dataset page reports 1,338 records and seven columns. [Dataset source](https://www.kaggle.com/datasets/mirichoi0218/insurance)

## Run the app

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Features
- Dataset preview and missing-value overview
- Numeric imputation and standardization
- Categorical imputation and one-hot encoding
- Linear Regression vs Ridge Regression comparison
- Interactive Ridge alpha and test split
- MAE, RMSE and R² metrics
- Actual vs predicted plot

## Structure

```text
RidgeRight/
├── app.py
├── data.csv
├── RidgeRight_Notebook.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

Dataset is not included in this project package. Download it manually and keep `data.csv` out of GitHub if you do not want to publish the raw dataset.
