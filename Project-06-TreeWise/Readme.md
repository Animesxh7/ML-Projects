# Project 06 — TreeWise 🌳

An interactive **Decision Tree Classifier** lab styled consistently with Project 05 (LogisticLens): matching typography, restrained teal palette, metric cards, sidebar controls and Plotly charts. using the UCI Red Wine Quality dataset (1,599 rows).

## What it demonstrates
- Classification from physicochemical wine measurements
- Binary target: original quality score **6+ = higher**; below 6 = lower
- Gini, entropy and log-loss split criteria
- Tree depth / leaf size controls and class balancing
- Accuracy, precision, recall, F1, ROC-AUC, confusion matrix and feature importance
- Interactive profile prediction and visual tree inspection

> Educational portfolio project. The binary target is a convenient teaching transformation, not an authoritative wine-quality assessment.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python download_data.py
streamlit run app.py
```

If you skip `download_data.py`, the app offers a dataset download button on first launch.

## Dataset
UCI Machine Learning Repository — [Wine Quality](https://archive.ics.uci.edu/dataset/ wine+quality) (red wine file). CSV is semicolon-separated.

## Files
- `app.py` — Streamlit concept demo
- `notebook.ipynb` — EDA, preprocessing, training and evaluation
- `download_data.py` — downloads the raw CSV
- `requirements.txt` — dependencies
- `data/` — dataset is downloaded here; raw data is not bundled
