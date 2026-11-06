from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="RidgeRight", page_icon="📈", layout="wide")
st.title("RidgeRight")
st.caption("Understanding Ridge Regression and L2 regularization")
st.write("Compare ordinary Linear Regression with Ridge Regression by changing the regularization strength.")
df = pd.read_csv('insurance.csv')
st.subheader("1. Dataset overview")
a, b, c = st.columns(3)
a.metric("Rows", f"{len(df):,}")
b.metric("Columns", len(df.columns))
c.metric("Missing values", int(df.isna().sum().sum()))
with st.expander("Preview dataset"):
    st.dataframe(df.head(10), use_container_width=True)
with st.expander("Column details"):
    st.dataframe(pd.DataFrame({
        "Data type": df.dtypes.astype(str),
        "Missing": df.isna().sum(),
        "Unique": df.nunique()
    }), use_container_width=True)

target = "charges"
if target not in df.columns:
    st.error("Expected target column 'charges'. Check that you downloaded the Medical Cost Personal Dataset.")
    st.write("Found columns:", list(df.columns))
    st.stop()

numeric = df.select_dtypes(include="number").columns.drop(target, errors="ignore").tolist()
categorical = df.select_dtypes(exclude="number").columns.tolist()
features = numeric + categorical

st.subheader("2. Configure the experiment")
left, right = st.columns(2)
with left:
    alpha = st.slider("Ridge alpha", 0.01, 100.0, 10.0, 0.01)
    test_percent = st.slider("Test data (%)", 10, 40, 20, 5)
with right:
    selected = st.multiselect("Input features", features, default=features)
    seed = st.number_input("Random state", 0, 9999, 42)

if not selected:
    st.warning("Select at least one feature.")
    st.stop()

num_sel = [x for x in selected if x in numeric]
cat_sel = [x for x in selected if x in categorical]
transformers = []
if num_sel:
    transformers.append(("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_sel))
if cat_sel:
    transformers.append(("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_sel))

X = df[selected].copy()
y = pd.to_numeric(df[target], errors="coerce")
valid = y.notna()
X, y = X.loc[valid], y.loc[valid]
if len(X) < 10:
    st.error("Not enough valid rows to train the model.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_percent / 100, random_state=int(seed)
)
prep = ColumnTransformer(transformers=transformers)
models = {
    "Linear Regression": Pipeline([("prep", prep), ("model", LinearRegression())]),
    "Ridge Regression": Pipeline([("prep", prep), ("model", Ridge(alpha=alpha))])
}
preds, scores = {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds[name] = model.predict(X_test)
    scores[name] = {
        "MAE": mean_absolute_error(y_test, preds[name]),
        "RMSE": mean_squared_error(y_test, preds[name]) ** 0.5,
        "R²": r2_score(y_test, preds[name])
    }

st.subheader("3. Model comparison")
st.dataframe(pd.DataFrame(scores).T.style.format("{:,.3f}"), use_container_width=True)
x1, x2, x3 = st.columns(3)
x1.metric("Ridge MAE", f"{scores['Ridge Regression']['MAE']:,.2f}")
x2.metric("Ridge RMSE", f"{scores['Ridge Regression']['RMSE']:,.2f}")
x3.metric("Ridge R²", f"{scores['Ridge Regression']['R²']:.3f}")

st.subheader("4. Actual vs predicted")
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_test, preds["Linear Regression"], alpha=.55, label="Linear")
ax.scatter(y_test, preds["Ridge Regression"], alpha=.55, label="Ridge")
lo = min(float(y_test.min()), *(float(p.min()) for p in preds.values()))
hi = max(float(y_test.max()), *(float(p.max()) for p in preds.values()))
ax.plot([lo, hi], [lo, hi], "--")
ax.set_xlabel("Actual insurance charges")
ax.set_ylabel("Predicted insurance charges")
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.subheader("5. Key learning")
st.markdown(
    f"- Current alpha: **{alpha:g}**.\\n"
    "- Ridge adds an L2 penalty that discourages very large coefficients.\\n"
    "- Compare the metrics to see whether regularization helps on this test split.\\n"
    "- Increasing alpha does not guarantee better performance; choose it using validation."
)
