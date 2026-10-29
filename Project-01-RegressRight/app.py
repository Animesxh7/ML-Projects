import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="RegressRight | Linear Regression",
    page_icon="📈",
    layout="wide"
)
sns.set_theme(style="whitegrid")
TARGET = "quality"

st.title("📈 RegressRight")
st.subheader("Linear Regression — From Data to Predictions")
st.write(
    "An end-to-end machine learning project using the "
    "UCI Red Wine Quality dataset."
)
# ---------------- LOAD AND CLEAN DATA ----------------
@st.cache_data
def load_data():
    raw = pd.read_csv("winequality-red.csv", sep=";")
    raw_rows = len(raw)

    df = raw.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    duplicate_count = int(df.duplicated().sum())
    missing_before = int(df.isna().sum().sum())

    df = df.drop_duplicates().copy()
    df = df.dropna(subset=[TARGET]).copy()

    return df, raw_rows, duplicate_count, missing_before

try:
    df, raw_rows, duplicate_count, missing_before = load_data()
except FileNotFoundError:
    st.error(
        "winequality-red.csv not found. Put it in the same folder "
        "as app.py."
    )
    st.stop()
except Exception as error:
    st.error("Could not load the dataset. Check the CSV format.")
    st.exception(error)
    st.stop()

if TARGET not in df.columns:
    st.error("The dataset must contain a 'quality' column.")
    st.stop()

if len(df) < 1000:
    st.error(f"Only {len(df)} rows remain after cleaning.")
    st.stop()

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Model Settings")

    test_size = st.slider(
        "Test set size (%)", 10, 40, 20, 5
    ) / 100

    random_state = st.slider(
        "Random seed", 0, 100, 42
    )

    st.divider()
    st.write("**Algorithm:** Linear Regression")
    st.write("**Target:** quality")
    st.write(f"**Features:** {X.shape[1]}")

# ---------------- TRAIN MODEL ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# ---------------- KPI METRICS ----------------
st.markdown("### 📌 Model Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Cleaned Rows", f"{len(df):,}")
c2.metric("Input Features", X.shape[1])
c3.metric("MAE", f"{mae:.3f}")
c4.metric("R² Score", f"{r2:.3f}")

st.caption(
    f"Training samples: {len(X_train):,} | "
    f"Test samples: {len(X_test):,}"
)

# ---------------- TABS ----------------
overview, eda, evaluation, prediction, insights = st.tabs([
    "🏠 Overview",
    "🔍 Explore Data",
    "📊 Evaluation",
    "🧪 Prediction Lab",
    "🧠 Model Insights"
])

# ================= OVERVIEW =================
with overview:
    st.subheader("Dataset Overview")

    a, b, c, d = st.columns(4)
    a.metric("Original Rows", f"{raw_rows:,}")
    b.metric("Original Columns", df.shape[1])
    c.metric("Duplicates Removed", duplicate_count)
    d.metric("Missing Values Before Cleaning", missing_before)

    st.markdown("#### First 10 Records")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("#### Features Used")
    st.write(", ".join(X.columns))

    st.markdown("#### Cleaning Process")
    st.markdown("""
    1. Loaded the semicolon-delimited CSV.
    2. Standardized column names.
    3. Converted columns to numeric values.
    4. Removed exact duplicate rows.
    5. Removed rows with a missing target.
    6. Median-imputed missing input values inside the model pipeline.
    """)

    st.info(
        "Imputation and scaling are fitted on training data only, "
        "helping prevent data leakage."
    )

# ================= EDA =================
with eda:
    st.subheader("Exploratory Data Analysis")

    feature = st.selectbox(
        "Choose a feature",
        X.columns.tolist(),
        index=(
            X.columns.tolist().index("alcohol")
            if "alcohol" in X.columns else 0
        )
    )

    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(df[feature], kde=True, ax=ax)
        ax.set_title(f"Distribution of {feature}")
        st.pyplot(fig)
        plt.close(fig)

    with right:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df, x=TARGET, y=feature, ax=ax)
        ax.set_title(f"{feature} by Quality")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Wine Quality Distribution")

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.countplot(data=df, x=TARGET, ax=ax)
    ax.set_title("Number of Wines by Quality Score")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Feature Correlation with Quality")

    correlations = (
        df.corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    correlations.plot(kind="barh", ax=ax)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Pearson Correlation")
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("View full correlation heatmap"):
        fig, ax = plt.subplots(figsize=(11, 8))
        sns.heatmap(
            df.corr(numeric_only=True),
            cmap="coolwarm",
            center=0,
            ax=ax
        )
        st.pyplot(fig)
        plt.close(fig)

# ================= EVALUATION =================
with evaluation:
    st.subheader("Evaluation on Unseen Test Data")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{mae:.4f}")
    m2.metric("MSE", f"{mse:.4f}")
    m3.metric("RMSE", f"{rmse:.4f}")
    m4.metric("R²", f"{r2:.4f}")

    st.markdown("""
    - **MAE:** Average absolute prediction error.
    - **MSE:** Average squared prediction error.
    - **RMSE:** Square root of MSE, in target units.
    - **R²:** Performance relative to predicting the test-set mean.
    """)

    left, right = st.columns(2)

    with left:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_test, y_pred, alpha=0.65)

        low = min(float(y_test.min()), float(y_pred.min()))
        high = max(float(y_test.max()), float(y_pred.max()))

        ax.plot(
            [low, high], [low, high],
            "r--", label="Perfect Predictions"
        )
        ax.set_xlabel("Actual Quality")
        ax.set_ylabel("Predicted Quality")
        ax.set_title("Actual vs Predicted")
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

    with right:
        residuals = y_test.to_numpy() - y_pred

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_pred, residuals, alpha=0.65)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Predicted Quality")
        ax.set_ylabel("Actual - Predicted")
        ax.set_title("Residual Plot")

        st.pyplot(fig)
        plt.close(fig)

    results = pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Predicted": y_pred,
        "Residual": y_test.to_numpy() - y_pred
    })

    results["Absolute Error"] = results["Residual"].abs()

    st.markdown("#### Largest Prediction Errors")
    st.dataframe(
        results.sort_values(
            "Absolute Error", ascending=False
        ).head(10).round(3),
        use_container_width=True
    )

    st.download_button(
        "Download Test Predictions CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="regressright_predictions.csv",
        mime="text/csv"
    )

# ================= PREDICTION LAB =================
with prediction:
    st.subheader("Predict Wine Quality")
    st.write(
        "Adjust the measurements to obtain a model-estimated "
        "wine quality score."
    )

    sample = {}
    input_columns = st.columns(3)

    for i, col in enumerate(X.columns):
        values = df[col].dropna()
        low = float(values.quantile(0.01))
        high = float(values.quantile(0.99))
        default = float(np.clip(values.median(), low, high))

        if low == high:
            low = float(values.min())
            high = float(values.max()) + 1e-6

        with input_columns[i % 3]:
            sample[col] = st.number_input(
                col.replace("_", " ").title(),
                min_value=low,
                max_value=high,
                value=default,
                format="%.4f"
            )

    if st.button("Predict Quality", type="primary"):
        sample_df = pd.DataFrame([sample], columns=X.columns)
        result = float(model.predict(sample_df)[0])

        st.success(f"Estimated Quality Score: {result:.2f}")
        st.caption(
            "This is a model estimate, not an official wine grade."
        )

# ================= MODEL INSIGHTS =================
with insights:
    st.subheader("How the Model Works")

    st.latex(
        r"\hat{y}=b_0+b_1x_1+b_2x_2+\cdots+b_px_p"
    )

    st.write(
        "Multiple Linear Regression estimates a numerical target "
        "from multiple input features."
    )

    regressor = model.named_steps["regressor"]

    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": regressor.coef_
    })

    coefficients["Magnitude"] = coefficients["Coefficient"].abs()
    coefficients = coefficients.sort_values(
        "Magnitude", ascending=False
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=coefficients,
        x="Coefficient",
        y="Feature",
        ax=ax
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Learned Feature Coefficients")
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        coefficients.drop(columns="Magnitude").round(4),
        use_container_width=True
    )

    st.markdown("""
    **Interpretation**

    - Positive coefficient: positive modeled association, holding
      other features constant.
    - Negative coefficient: negative modeled association.
    - Standardization means coefficients correspond to a
      one-standard-deviation change in an input feature.
    """)

    st.warning(
        "Wine quality is an ordinal sensory score. Linear Regression "
        "is an educational baseline, not a definitive wine-grading "
        "system. Coefficients and correlations do not prove causation."
    )

st.divider()
st.caption(
    "RegressRight | Traditional ML | Dataset: UCI Red Wine Quality"
)