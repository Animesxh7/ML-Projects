# 🍷 RegressRight — Wine Quality Prediction Using Linear Regression

An end-to-end Machine Learning project that predicts wine quality scores from physicochemical properties using **Linear Regression**. The project includes data cleaning, exploratory data analysis (EDA), preprocessing, model evaluation, and an interactive Streamlit application.

## 📌 Project Overview

Wine quality depends on several measurable properties, such as acidity, residual sugar, chlorides, sulphates, and alcohol content. RegressRight explores how these features relate to the quality score and builds a regression model to estimate wine quality.

**Project highlights**
- End-to-end supervised Machine Learning workflow
- Data inspection, cleaning, and exploratory data analysis
- Missing-value handling and feature scaling through a Scikit-learn pipeline
- Linear Regression model training and evaluation
- Actual-versus-predicted and residual visualizations
- Interactive Streamlit dashboard with a prediction interface

## 🎯 Problem Statement

Build a regression model that estimates a red wine's quality score using its physicochemical measurements.

- **Problem type:** Supervised Learning
- **ML task:** Regression
- **Algorithm:** Linear Regression
- **Target variable:** `quality`
- **Dataset:** UCI Red Wine Quality Dataset

## 📂 Project Structure

```text
RegressRight/
├── app.py
├── winequality-red.csv
├── RegressRight_Linear_Regression.ipynb
├── requirements.txt
├── README.md
└── LICENSE_DATASET.txt
```

## 📊 Dataset

The project uses the red wine subset of the Wine Quality dataset. The original dataset contains **1,599 observations**, with 11 physicochemical input features and one quality target.

The CSV file uses a **semicolon (`;`) delimiter**, so it must be loaded accordingly.

### Features

| Feature | Description |
|---|---|
| `fixed acidity` | Non-volatile acids in wine |
| `volatile acidity` | Volatile acids in wine |
| `citric acid` | Citric acid content |
| `residual sugar` | Sugar remaining after fermentation |
| `chlorides` | Salt content |
| `free sulfur dioxide` | Free SO₂ concentration |
| `total sulfur dioxide` | Total SO₂ concentration |
| `density` | Wine density |
| `pH` | Acidity/alkalinity measure |
| `sulphates` | Sulphate content |
| `alcohol` | Alcohol percentage |
| `quality` | Target quality score |

## ⚙️ Machine Learning Workflow

1. **Data loading:** Load the semicolon-delimited CSV file using Pandas.
2. **Data inspection:** Examine dimensions, column types, descriptive statistics, and missing values.
3. **Data cleaning:** Identify and handle duplicate records as implemented in the notebook.
4. **EDA:** Explore distributions, feature relationships, correlations, and quality-score patterns.
5. **Feature and target separation:** Use the 11 physicochemical measurements as features (`X`) and `quality` as the target (`y`).
6. **Train-test split:** Separate data into training and testing sets.
7. **Preprocessing:** Use median imputation and standardization in a Scikit-learn pipeline.
8. **Model training:** Fit a Linear Regression estimator on the training data.
9. **Evaluation:** Measure performance on held-out test data.
10. **Visualization and prediction:** Explore model results and make predictions through the Streamlit app.

## 📈 Model Evaluation

The model is evaluated using regression metrics:

| Metric | What it measures |
|---|---|
| MAE | Average absolute prediction error |
| MSE | Mean squared prediction error |
| RMSE | Prediction error in the target's units |
| R² Score | Proportion of target variance explained by the model |

Actual metric values depend on the train-test split and the preprocessing and cleaning steps. Run the notebook to obtain the results for your current version of the dataset.

## 🖥️ Interactive Streamlit App

The Streamlit application provides an interactive way to explore the dataset and understand the trained model.

Depending on the implemented app version, its sections include:

- **Overview:** Project summary and dataset/model information
- **Explore Data:** Dataset preview and exploratory visualizations
- **Evaluation:** Regression metrics and model performance plots
- **Prediction Lab:** Enter feature values to estimate wine quality
- **Model Insights:** Explore fitted coefficients and feature relationships

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RegressRight
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your actual GitHub repository URL.

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit application

Ensure that `app.py`, `requirements.txt`, and `winequality-red.csv` are in the project folder. Then run:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

### 5. Run the Jupyter Notebook

Launch Jupyter:

```bash
python -m pip install notebook
python -m notebook
```

Open `RegressRight_Linear_Regression.ipynb` and execute the cells in order.

## 🛠️ Tech Stack

- **Language:** Python
- **Data manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Interactive application:** Streamlit
- **Development environment:** Jupyter Notebook / PyCharm

## 📚 What I Learned

- Building an end-to-end regression workflow
- Exploring and cleaning real-world tabular data
- Preventing preprocessing leakage by fitting transformations within a pipeline
- Evaluating regression models using multiple metrics
- Communicating model performance through visualizations
- Building an interactive ML demonstration with Streamlit

## ⚠️ Limitations

- Linear Regression assumes a linear relationship between input features and the target.
- Wine quality scores are subjective and may not be fully explained by physicochemical measurements.
- Predictions are estimates, not certified assessments of wine quality.
- Model performance should be interpreted using held-out evaluation results rather than training performance alone.

## 📄 Dataset Attribution

The dataset originates from the UCI Machine Learning Repository's Wine Quality dataset.

Dataset reference: https://archive.ics.uci.edu/dataset/186/wine+quality

Please review the dataset's original licensing and attribution requirements before redistribution. Refer to `LICENSE_DATASET.txt` if included in the repository.
⭐ If you find this project useful, consider starring the repository.
