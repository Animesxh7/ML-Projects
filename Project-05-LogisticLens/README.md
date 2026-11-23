# LogisticLens — Project 05

A small Streamlit concept demo for **Logistic Regression**, a supervised binary-classification algorithm.

## Objective
Predict whether a bank customer subscribes to a term deposit (`yes` / `no`) using the UCI Bank Marketing dataset.

## Dataset
- Source: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- File: `bank-full.csv` (downloaded locally as `data.csv`)
- 45,211 rows and 16 input features plus the target
- The app uses an 80/20 stratified train/test split.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python download_data.py
streamlit run app.py
```

## What the demo covers
- Sigmoid function and predicted probabilities
- Numeric scaling and categorical one-hot encoding
- Logistic Regression with balanced class weights
- Accuracy, precision, recall, F1-score and ROC-AUC
- Confusion matrix, ROC curve and interactive single-customer prediction

## Important note
The `duration` feature is known only after a call has taken place. It is included for a teaching demonstration and should be excluded when designing a model intended to decide whom to call before a campaign contact.

## Project structure

```text
Project-05-LogisticLens/
├── app.py
├── download_data.py
├── requirements.txt
├── README.md
└── .gitignore
```
