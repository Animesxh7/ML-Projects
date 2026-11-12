import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="LassoRight", page_icon="📉", layout="wide")

FEATURES = [
    "cement", "blast_furnace_slag", "fly_ash", "water", "superplasticizer",
    "coarse_aggregate", "fine_aggregate", "age"
]
TARGET = "concrete_strength"

st.title("LassoRight")
st.subheader("Explore Lasso Regression (L1 regularization)")
st.write(
    "Lasso Regression adds an L1 penalty to the regression loss. "
    "This shrinks coefficients, and some can become exactly zero, "
    "which makes Lasso useful for feature selection."
)

with st.expander("What does alpha do?"):
    st.markdown(
        "- **Small alpha:** weaker regularization; coefficients are less constrained.\n"
        "- **Large alpha:** stronger regularization; more coefficients may become zero.\n"
        "- Features are standardized before fitting so the penalty is applied fairly."
    )

try:
    df = pd.read_csv("Concrete_Data.csv")
except FileNotFoundError:
    st.error("Dataset 'Concrete_Data.csv' not found. Please ensure the file is in the same directory.")
    st.stop()

if len(df.columns) == 9:
    df.columns = FEATURES + [TARGET]
else:
    st.error(f"Expected 9 columns in Concrete_Data.csv, but found {len(df.columns)}.")
    st.stop()

df = df[FEATURES + [TARGET]].dropna()
X = df[FEATURES]
y = df[TARGET]

with st.sidebar:
    st.header("Model settings")
    alpha = st.slider("Lasso alpha", min_value=0.001, max_value=10.0, value=0.1, step=0.001)
    test_size = st.slider("Test set size (%)", 10, 40, 20, 5) / 100
    random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=int(random_state)
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(alpha=alpha, max_iter=20000))
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

st.markdown("### Model evaluation")
m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:.2f} MPa")
m2.metric("RMSE", f"{rmse:.2f} MPa")
m3.metric("R² score", f"{r2:.3f}")

st.caption("Metrics are calculated on the held-out test set. They update when model settings change.")

st.markdown("### Actual vs predicted")
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_test, predictions, alpha=0.65)
low = min(float(y_test.min()), float(predictions.min()))
high = max(float(y_test.max()), float(predictions.max()))
ax.plot([low, high], [low, high], linestyle="--")
ax.set_xlabel("Actual strength (MPa)")
ax.set_ylabel("Predicted strength (MPa)")
ax.set_title("Actual vs predicted concrete strength")
st.pyplot(fig)
plt.close(fig)

st.markdown("### Lasso coefficients")
coefs = pd.Series(model.named_steps["lasso"].coef_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
coefs.plot(kind="barh", ax=ax)
ax.axvline(0, linewidth=0.8)
ax.set_xlabel("Coefficient (standardized feature scale)")
ax.set_title("Lasso coefficients")
st.pyplot(fig)
plt.close(fig)
zero_count = int(np.isclose(coefs.values, 0.0, atol=1e-8).sum())
st.write(f"**Features with coefficients shrunk to zero:** {zero_count} of {len(FEATURES)}")

st.markdown("### Try a prediction")
st.write("Enter a concrete mix to estimate its compressive strength.")

defaults = {
    "cement": 300.0,
    "blast_furnace_slag": 100.0,
    "fly_ash": 0.0,
    "water": 180.0,
    "superplasticizer": 8.0,
    "coarse_aggregate": 1000.0,
    "fine_aggregate": 700.0,
    "age": 28.0,
}
ranges = {
    "cement": (0.0, 600.0),
    "blast_furnace_slag": (0.0, 400.0),
    "fly_ash": (0.0, 250.0),
    "water": (100.0, 250.0),
    "superplasticizer": (0.0, 35.0),
    "coarse_aggregate": (800.0, 1200.0),
    "fine_aggregate": (500.0, 1000.0),
    "age": (1.0, 365.0),
}
input_values = {}
left, right = st.columns(2)
for i, feature in enumerate(FEATURES):
    parent = left if i % 2 == 0 else right
    minimum, maximum = ranges[feature]
    with parent:
        input_values[feature] = st.number_input(
            f"{feature.replace('_', ' ').title()}",
            min_value=float(minimum),
            max_value=float(maximum),
            value=float(defaults[feature]),
            step=1.0,
        )

if st.button("Predict concrete strength", type="primary"):
    input_frame = pd.DataFrame([input_values], columns=FEATURES)
    result = float(model.predict(input_frame)[0])
    st.success(f"Estimated compressive strength: **{result:.2f} MPa**")
    st.caption("This is a demonstration model estimate, not an engineering certification.")

with st.expander("Dataset preview"):
    st.write(f"Rows: {len(df):,} | Features: {len(FEATURES)}")
    st.dataframe(df.head(10), use_container_width=True)