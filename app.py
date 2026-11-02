
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PolyPredict | Polynomial Regression",
    page_icon="◈",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown("""
<style>
    .block-container {
        max-width: 1120px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.5px;
    }

    .subtitle {
        color: #89919d;
        font-size: 15px;
        margin-top: -12px;
        margin-bottom: 30px;
    }

    .section-label {
        color: #89919d;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: #151c27;
        border: 1px solid #283242;
        padding: 18px;
        border-radius: 10px;
    }

    div[data-testid="stMetricLabel"] {
        color: #aab3c0;
    }

    .project-footer {
        color: #89919d;
        font-size: 13px;
        text-align: center;
        padding-top: 30px;
    }

    hr {
        border-color: #283242;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("# ◈ PolyPredict")
st.markdown(
    '<p class="subtitle">Understanding non-linear relationships through Polynomial Regression.</p>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# DATASET LOADING
# --------------------------------------------------

@st.cache_data
def load_dataset(file):
    return pd.read_csv(file, sep=";")


dataset_path = Path(__file__).parent / "winequality-white.csv"

st.sidebar.title("Project settings")
st.sidebar.caption("Dataset & model configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload wine quality CSV",
    type=["csv"]
)

try:
    if uploaded_file is not None:
        df = load_dataset(uploaded_file)
        dataset_source = "Uploaded dataset"

    elif dataset_path.exists():
        df = load_dataset(dataset_path)
        dataset_source = "Local dataset"

    else:
        st.warning(
            "Dataset not found. Upload winequality-white.csv "
            "using the sidebar to continue."
        )
        st.stop()

except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()


# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

required_columns = [
    "alcohol",
    "volatile acidity",
    "sulphates",
    "quality"
]

if not all(col in df.columns for col in required_columns):
    st.error("The uploaded CSV does not contain the required columns.")
    st.stop()

df = df[required_columns].copy()
df = df.drop_duplicates()
df = df.dropna()

features = ["alcohol", "volatile acidity", "sulphates"]
target = "quality"

X = df[features]
y = df[target]


# --------------------------------------------------
# SIDEBAR MODEL CONTROLS
# --------------------------------------------------

st.sidebar.divider()

st.sidebar.subheader("Model configuration")

degree = st.sidebar.selectbox(
    "Polynomial degree",
    options=[2, 3, 4],
    index=0
)

test_size = st.sidebar.slider(
    "Test data proportion",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05
)

st.sidebar.caption(
    "Higher polynomial degrees allow more complex relationships."
)

# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42
)

linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

poly_model = Pipeline([
    ("polynomial", PolynomialFeatures(
        degree=degree,
        include_bias=False
    )),
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)
poly_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)
poly_predictions = poly_model.predict(X_test)


# --------------------------------------------------
# METRICS
# --------------------------------------------------

linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_rmse = np.sqrt(mean_squared_error(y_test, linear_predictions))
linear_r2 = r2_score(y_test, linear_predictions)

poly_mae = mean_absolute_error(y_test, poly_predictions)
poly_rmse = np.sqrt(mean_squared_error(y_test, poly_predictions))
poly_r2 = r2_score(y_test, poly_predictions)


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

st.markdown('<p class="section-label">01 / Dataset overview</p>',
            unsafe_allow_html=True)

st.subheader("Wine quality analysis")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total records", f"{len(df):,}")
col2.metric("Input features", len(features))
col3.metric("Training records", f"{len(X_train):,}")
col4.metric("Testing records", f"{len(X_test):,}")

st.caption(f"Dataset source: {dataset_source}")

with st.expander("Explore dataset"):
    st.dataframe(df.head(10), use_container_width=True)
    st.write("Dataset summary")
    st.dataframe(df.describe().round(2), use_container_width=True)


st.divider()


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.markdown('<p class="section-label">02 / Model performance</p>',
            unsafe_allow_html=True)

st.subheader("Linear vs Polynomial Regression")

st.caption(
    f"Polynomial Regression is currently using degree {degree}."
)

metric_col1, metric_col2, metric_col3 = st.columns(3)

metric_col1.metric(
    "Polynomial MAE",
    f"{poly_mae:.3f}",
    delta=f"{linear_mae - poly_mae:.3f} vs Linear",
    delta_color="normal"
)

metric_col2.metric(
    "Polynomial RMSE",
    f"{poly_rmse:.3f}",
    delta=f"{linear_rmse - poly_rmse:.3f} vs Linear",
    delta_color="normal"
)

metric_col3.metric(
    "Polynomial R²",
    f"{poly_r2:.3f}",
    delta=f"{poly_r2 - linear_r2:.3f} vs Linear",
    delta_color="normal"
)


# --------------------------------------------------
# COMPARISON TABLE
# --------------------------------------------------

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Polynomial Regression"],
    "MAE": [linear_mae, poly_mae],
    "RMSE": [linear_rmse, poly_rmse],
    "R² Score": [linear_r2, poly_r2]
})

st.markdown("#### Performance comparison")
st.dataframe(
    comparison.round(4),
    hide_index=True,
    use_container_width=True
)


# --------------------------------------------------
# ACTUAL VS PREDICTED
# --------------------------------------------------

st.divider()

st.markdown('<p class="section-label">03 / Visual analysis</p>',
            unsafe_allow_html=True)

st.subheader("Actual vs predicted quality")

fig, ax = plt.subplots(figsize=(9, 5))

ax.scatter(
    y_test,
    poly_predictions,
    alpha=0.55,
    s=35
)

min_value = min(y_test.min(), poly_predictions.min())
max_value = max(y_test.max(), poly_predictions.max())

ax.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=1.5
)

ax.set_xlabel("Actual quality")
ax.set_ylabel("Predicted quality")
ax.set_title("Polynomial Regression — Prediction Analysis")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# RESIDUAL ANALYSIS
# --------------------------------------------------

st.subheader("Residual distribution")

residuals = y_test - poly_predictions

fig, ax = plt.subplots(figsize=(9, 4))

ax.hist(
    residuals,
    bins=25,
    edgecolor="white",
    alpha=0.8
)

ax.axvline(0, linestyle="--", linewidth=1.5)
ax.set_xlabel("Residual")
ax.set_ylabel("Frequency")
ax.set_title("Prediction error distribution")

fig.tight_layout()
st.pyplot(fig)
plt.close(fig)


# --------------------------------------------------
# INTERACTIVE PREDICTION
# --------------------------------------------------

st.divider()

st.markdown('<p class="section-label">04 / Try the model</p>',
            unsafe_allow_html=True)

st.subheader("Predict wine quality")

st.caption(
    "Enter three wine characteristics to generate a quality prediction."
)

input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    alcohol = st.number_input(
        "Alcohol (%)",
        min_value=0.0,
        max_value=20.0,
        value=10.5,
        step=0.1
    )

with input_col2:
    volatile_acidity = st.number_input(
        "Volatile acidity",
        min_value=0.0,
        max_value=2.0,
        value=0.30,
        step=0.01
    )

with input_col3:
    sulphates = st.number_input(
        "Sulphates",
        min_value=0.0,
        max_value=2.0,
        value=0.45,
        step=0.01
    )


if st.button("Predict wine quality", type="primary", use_container_width=True):

    sample = pd.DataFrame({
        "alcohol": [alcohol],
        "volatile acidity": [volatile_acidity],
        "sulphates": [sulphates]
    })

    prediction = poly_model.predict(sample)[0]

    st.success("Prediction generated successfully.")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Predicted quality",
            f"{prediction:.2f} / 10"
        )

    with result_col2:
        st.metric(
            "Nearest quality score",
            f"{int(np.clip(np.rint(prediction), 0, 10))} / 10"
        )

    st.caption(
        "This is a model-based estimate, not a professional wine assessment."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown(
    """
    <div class="project-footer">
        PolyPredict · Polynomial Regression<br>
        Built with Python, Scikit-learn, Pandas & Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
