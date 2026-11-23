"""LogisticLens: an interactive Logistic Regression lab on the UCI Bank Marketing data."""
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="LogisticLens", page_icon="🎯", layout="wide")

DATA_PATH = Path(__file__).with_name("data.csv")
DATA_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

INK, TEAL, ROSE, AMBER, SLATE, GRID = "#16243A", "#0E7C86", "#CF5C6E", "#E3A63B", "#8793A5", "#E3E8EF"

FIELD_LABELS = {
    "age": "Age", "job": "Job", "marital": "Marital status", "education": "Education",
    "default": "Credit in default", "balance": "Account balance (€)", "housing": "Housing loan",
    "loan": "Personal loan", "contact": "Contact type", "day": "Day of month",
    "month": "Last contact month", "duration": "Last call duration (seconds)",
    "campaign": "Contacts this campaign", "pdays": "Days since previous campaign (-1 = never)",
    "previous": "Contacts before this campaign", "poutcome": "Previous campaign outcome",
}

PRESETS = {
    "Typical customer": {},
    "Student, long call": {"age": 22, "job": "student", "marital": "single", "education": "tertiary",
                           "balance": 800, "duration": 420},
    "Retiree with savings": {"age": 67, "job": "retired", "marital": "married", "education": "secondary",
                             "balance": 6000, "housing": "no", "loan": "no", "contact": "cellular", "duration": 300},
    "Busy, short call": {"age": 38, "job": "management", "education": "tertiary", "balance": 2500,
                         "housing": "yes", "loan": "yes", "duration": 60, "campaign": 6},
    "Returning customer": {"poutcome": "success", "previous": 2, "pdays": 90, "duration": 250,
                                "contact": "cellular"},
}

MODEL_DEFAULTS = {"s_C": 1.0, "s_balanced": True, "s_test": 20, "s_drop": False, "s_thr": 0.50}

# ----------------------------------------------------------------------------- styling
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700&family=Figtree:wght@400;500;600&display=swap');
html, body, .stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp textarea, .stApp button {{ font-family: 'Figtree', system-ui, sans-serif; }}
[data-testid="stIconMaterial"], [data-testid="stIconMaterial"] * {{ font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important; }}
h1, h2, h3 {{ font-family: 'Bricolage Grotesque', 'Figtree', sans-serif; letter-spacing: -0.01em; color: {INK}; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1280px; }}
.hero h1 {{ font-size: 2.6rem; margin: 0 0 .15rem 0; padding: 0; }}
.hero p {{ font-size: 1.05rem; color: #4A5A70; margin: 0 0 .4rem 0; max-width: 62ch; }}
.status {{ color: #5B6B80; font-size: .9rem; margin-bottom: 1rem; }}
[data-testid="stMetric"] {{ background: #fff; border: 1px solid {GRID}; border-radius: 10px; padding: 14px 16px; }}
[data-testid="stMetricLabel"] p {{ color: #5B6B80; font-size: .85rem; }}
button[data-baseweb="tab"] {{ font-weight: 600; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {GRID}; }}
.verdict {{ background: #fff; border: 1px solid {GRID}; border-radius: 14px; padding: 22px 24px 18px; }}
.verdict .cap {{ color: #5B6B80; font-size: .9rem; }}
.verdict .num {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 4.2rem; line-height: 1.05; font-weight: 700; color: {INK}; }}
.verdict .lab {{ font-weight: 600; margin: .1rem 0 1rem; }}
.verdict .lab.yes {{ color: {TEAL}; }} .verdict .lab.no {{ color: {ROSE}; }}
.meter {{ position: relative; height: 12px; background: {GRID}; border-radius: 6px; }}
.meter .fill {{ position: absolute; inset: 0 auto 0 0; border-radius: 6px; }}
.meter .tick {{ position: absolute; top: -5px; width: 3px; height: 22px; background: {AMBER}; border-radius: 2px; }}
.meter-scale {{ display: flex; justify-content: space-between; color: #5B6B80; font-size: .8rem; margin-top: 8px; }}
.callout {{ background: #fff; border: 1px solid {GRID}; border-left: 4px solid {TEAL}; border-radius: 8px; padding: 12px 16px; margin: .4rem 0 1rem; }}
.callout.warn {{ border-left-color: {AMBER}; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- helpers
def show(fig, height=340, title=None):
    """Apply the shared chart style and render a Plotly figure."""
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=70 if title else 40, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Figtree, sans-serif", color=INK, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=1, xanchor="right"),
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)) if title else None,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:  # older Streamlit releases
        st.plotly_chart(fig, use_container_width=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def callout(text, warn=False):
    st.markdown(f'<div class="callout{" warn" if warn else ""}">{text}</div>', unsafe_allow_html=True)


def fetch_dataset():
    """Download bank-full.csv from UCI and save it as data.csv."""
    with urlopen(DATA_URL, timeout=60) as response:
        outer_bytes = response.read()
    with ZipFile(BytesIO(outer_bytes)) as outer:
        inner_name = next((n for n in outer.namelist() if n.lower().endswith("bank.zip")), None)
        if inner_name is None:
            raise RuntimeError("Could not find bank.zip inside the UCI archive.")
        inner_bytes = outer.read(inner_name)
    with ZipFile(BytesIO(inner_bytes)) as inner:
        csv_name = next((n for n in inner.namelist() if n.lower().endswith("bank-full.csv")), None)
        if csv_name is None:
            raise RuntimeError("Could not find bank-full.csv inside bank.zip.")
        DATA_PATH.write_bytes(inner.read(csv_name))


@st.cache_data(show_spinner=False)
def load_data(path, mtime):
    return pd.read_csv(path, sep=";")


@st.cache_data(show_spinner=False)
def build_profile(_df, key):
    """Typical values and allowed options for every input field."""
    X = _df.drop(columns="y")
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    defaults, options, ranges = {}, {}, {}
    for c in num:
        defaults[c] = int(round(X[c].median()))
        ranges[c] = (float(X[c].quantile(0.01)), float(X[c].quantile(0.99)))
    for c in cat:
        defaults[c] = X[c].mode().iloc[0]
        values = sorted(X[c].unique())
        options[c] = [m for m in MONTHS if m in values] if c == "month" else values
    return {"defaults": defaults, "options": options, "ranges": ranges, "numeric": num, "categorical": cat}


def owner_of(name, categorical):
    """Map a transformed feature name back to the original column."""
    raw = name.split("__", 1)[1]
    if name.startswith("cat__"):
        matches = [c for c in categorical if raw.startswith(c + "_")]
        return max(matches, key=len)
    return raw


@st.cache_resource(show_spinner=False, max_entries=8)
def train_model(_df, data_key, C, balanced, test_size, drop_duration):
    X = _df.drop(columns="y")
    if drop_duration:
        X = X.drop(columns="duration")
    y = (_df["y"] == "yes").astype(int)
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(
            C=C, max_iter=1000, class_weight="balanced" if balanced else None, random_state=42)),
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    Xt = preprocess.transform(X_train)
    train_mean = np.asarray(Xt.mean(axis=0)).ravel()
    names = preprocess.get_feature_names_out()
    return {
        "model": model, "X_test": X_test, "y_test": y_test.to_numpy(), "proba": proba,
        "numeric": numeric, "categorical": categorical, "feature_cols": X.columns.tolist(),
        "train_mean": train_mean, "names": list(names),
        "owners": [owner_of(n, categorical) for n in names], "n_train": len(X_train),
    }


def explain(art, sample):
    """Log-odds shift each field causes relative to an average customer."""
    pre = art["model"].named_steps["preprocess"]
    coefs = art["model"].named_steps["classifier"].coef_[0]
    intercept = art["model"].named_steps["classifier"].intercept_[0]
    x = pre.transform(sample)
    x = x.toarray() if sparse.issparse(x) else np.asarray(x)
    delta = (x.ravel() - art["train_mean"]) * coefs
    by_field = {}
    for owner, d in zip(art["owners"], delta):
        by_field[owner] = by_field.get(owner, 0.0) + float(d)
    baseline = float(intercept + art["train_mean"] @ coefs)
    return baseline, by_field


def readable_feature(name, art):
    raw = name.split("__", 1)[1]
    if name.startswith("cat__"):
        col = owner_of(name, art["categorical"])
        return f"{col} = {raw[len(col) + 1:]}"
    return f"{raw} (+1 std dev)"


def reset_model_settings():
    for k, v in MODEL_DEFAULTS.items():
        st.session_state[k] = v


def apply_preset(name, profile):
    values = {**profile["defaults"], **PRESETS[name]}
    for col, val in values.items():
        if col in profile["options"] and val not in profile["options"][col]:
            continue
        st.session_state[f"p_{col}"] = val


# ----------------------------------------------------------------------------- header
st.markdown("""
<div class="hero">
  <h1>LogisticLens</h1>
  <p>See how a logistic regression decides which bank customers will subscribe to a term deposit.
  Change the model, the threshold or a customer and watch the result move.</p>
</div>
""", unsafe_allow_html=True)

if not DATA_PATH.exists():
    callout("The dataset isn't on this machine yet. Download the UCI Bank Marketing data (about 5 MB) to continue.", warn=True)
    if st.button("Download dataset", type="primary"):
        try:
            with st.spinner("Downloading from the UCI repository..."):
                fetch_dataset()
            st.rerun()
        except Exception as exc:  # network or archive problems
            st.error(f"Download failed: {exc}. Run `python download_data.py` or place `data.csv` next to `app.py`.")
    st.stop()

mtime = DATA_PATH.stat().st_mtime
df = load_data(str(DATA_PATH), mtime)
profile = build_profile(df, mtime)
for col, val in profile["defaults"].items():
    st.session_state.setdefault(f"p_{col}", val)
for k, v in MODEL_DEFAULTS.items():
    st.session_state.setdefault(k, v)

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Model settings")
    st.select_slider(
        "Regularization strength (C)", options=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0], key="s_C",
        help="Smaller C means stronger regularization: coefficients are pulled toward zero and the model is simpler.",
    )
    st.toggle("Balance the classes", key="s_balanced",
              help="Weights the rare 'yes' class more heavily so the model doesn't just predict 'no' for everyone.")
    st.slider("Test set size (%)", 10, 40, step=5, key="s_test",
              help="Share of customers held back to evaluate the model.")
    st.toggle("Leave out call duration", key="s_drop",
              help="Duration is only known after the call, so a model that decides whom to call can't use it.")
    if not st.session_state["s_drop"]:
        st.caption("Call duration is included. It's known only after a call, so scores look better than they would in a real campaign.")
    st.markdown("### Decision threshold")
    st.slider("Predict 'yes' when probability is at least", 0.05, 0.95, step=0.01, key="s_thr",
              help="Lower it to find more subscribers at the cost of more wasted calls. Raise it for fewer, surer calls.")
    st.button("Reset settings", on_click=reset_model_settings)

with st.spinner("Training the model..."):
    art = train_model(df, mtime, st.session_state["s_C"], st.session_state["s_balanced"],
                      st.session_state["s_test"] / 100, st.session_state["s_drop"])

thr = st.session_state["s_thr"]
y_test, proba = art["y_test"], art["proba"]
pred = (proba >= thr).astype(int)
auc = roc_auc_score(y_test, proba)

st.markdown(
    f'<div class="status">Trained on {art["n_train"]:,} customers and tested on {len(y_test):,}. '
    f'ROC-AUC {auc:.3f} · threshold {thr:.2f}</div>', unsafe_allow_html=True)

tab_explore, tab_sigmoid, tab_perf, tab_drivers, tab_predict, tab_concepts = st.tabs(
    ["Explore data", "Sigmoid lab", "Model performance", "What drives it", "Predict", "Key concepts"])

# ----------------------------------------------------------------------------- explore
with tab_explore:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{len(df):,}")
    k2.metric("Input features", str(df.shape[1] - 1))
    k3.metric("Subscribed", f"{(df['y'] == 'yes').mean():.1%}")
    k4.metric("Test customers", f"{len(y_test):,}")

    st.markdown("#### How does each field relate to subscribing?")
    feature = st.selectbox("Field", df.columns.drop("y"), format_func=lambda c: FIELD_LABELS.get(c, c))
    overall = (df["y"] == "yes").mean()
    is_numeric = feature in profile["numeric"]

    if is_numeric:
        view = st.radio("Show", ["Subscription rate", "Distribution"], horizontal=True)
        lo, hi = profile["ranges"][feature]
        if view == "Distribution":
            fig = go.Figure()
            clipped = df[(df[feature] >= lo) & (df[feature] <= hi)]
            for label, colour in (("no", SLATE), ("yes", TEAL)):
                fig.add_histogram(x=clipped.loc[clipped["y"] == label, feature], name=f"Subscribed: {label}",
                                  marker_color=colour, opacity=0.65, histnorm="percent", nbinsx=40)
            fig.update_layout(barmode="overlay", xaxis_title=FIELD_LABELS.get(feature, feature),
                              yaxis_title="% of group")
            show(fig, 380)
            st.caption("Each group is scaled to 100% so the rare 'yes' group stays visible. The axis covers the 1st to 99th percentile.")
        else:
            bins = pd.qcut(df[feature], q=10, duplicates="drop")
            grouped = df.groupby(bins, observed=True).agg(rate=("y", lambda s: (s == "yes").mean()), n=("y", "size"))
            labels = [f"{iv.left:,.0f} to {iv.right:,.0f}" for iv in grouped.index]
            fig = go.Figure(go.Bar(x=labels, y=grouped["rate"], marker_color=TEAL, customdata=grouped["n"],
                                   hovertemplate="%{x}<br>%{y:.1%} subscribed<br>%{customdata:,} customers<extra></extra>"))
            fig.add_hline(y=overall, line_dash="dot", line_color=AMBER,
                          annotation_text=f"Overall {overall:.1%}", annotation_position="top left")
            fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Subscription rate",
                              xaxis_title=f"{FIELD_LABELS.get(feature, feature)} (grouped into deciles)")
            show(fig, 380)
    else:
        grouped = df.groupby(feature).agg(rate=("y", lambda s: (s == "yes").mean()), n=("y", "size"))
        grouped = grouped.reindex([m for m in MONTHS if m in grouped.index]) if feature == "month" \
            else grouped.sort_values("rate", ascending=False)
        fig = go.Figure(go.Bar(x=grouped.index.astype(str), y=grouped["rate"], marker_color=TEAL,
                               customdata=grouped["n"], text=[f"{r:.0%}" for r in grouped["rate"]],
                               textposition="outside",
                               hovertemplate="%{x}<br>%{y:.1%} subscribed<br>%{customdata:,} customers<extra></extra>"))
        fig.add_hline(y=overall, line_dash="dot", line_color=AMBER,
                      annotation_text=f"Overall {overall:.1%}", annotation_position="top right")
        fig.update_layout(yaxis_tickformat=".0%", yaxis_title="Subscription rate",
                          xaxis_title=FIELD_LABELS.get(feature, feature))
        show(fig, 380)

    with st.expander("Browse the raw data"):
        st.dataframe(df.sample(300, random_state=1).reset_index(drop=True), width="stretch", height=320)

# ----------------------------------------------------------------------------- sigmoid lab
with tab_sigmoid:
    st.markdown("#### From a score to a probability")
    st.write("A logistic regression adds up weighted inputs into a score *z*, then the sigmoid squeezes it into a probability. "
             "Move the sliders to see how the curve and the decision boundary change.")
    c_left, c_right = st.columns([1, 2])
    with c_left:
        b0 = st.slider("Intercept β₀ (shifts the curve)", -6.0, 6.0, 0.0, 0.1)
        b1 = st.slider("Weight β₁ (steepness and direction)", -3.0, 3.0, 1.0, 0.1)
        x0 = st.slider("Feature value x", -10.0, 10.0, 2.0, 0.1)
        z0 = b0 + b1 * x0
        p0 = float(sigmoid(z0))
        m1, m2 = st.columns(2)
        m1.metric("Score z", f"{z0:.2f}")
        m2.metric("Probability", f"{p0:.1%}")
        st.metric("Odds of 'yes'", f"{np.exp(z0):,.2f} to 1" if z0 < 12 else "very high")
    with c_right:
        xs = np.linspace(-10, 10, 400)
        fig = go.Figure()
        fig.add_scatter(x=xs, y=sigmoid(b0 + b1 * xs), mode="lines", line=dict(color=TEAL, width=3), name="P(y=1)")
        fig.add_hline(y=thr, line_dash="dot", line_color=AMBER,
                      annotation_text=f"Threshold {thr:.2f}", annotation_position="bottom right")
        if abs(b1) > 1e-9:
            boundary = (np.log(thr / (1 - thr)) - b0) / b1
            if -10 <= boundary <= 10:
                fig.add_vline(x=boundary, line_dash="dash", line_color=SLATE,
                              annotation_text=f"Boundary x = {boundary:.2f}", annotation_position="top")
        fig.add_scatter(x=[x0], y=[p0], mode="markers", marker=dict(size=14, color=INK, line=dict(color="#fff", width=2)),
                        name="Your point", hovertemplate="x = %{x:.1f}<br>P = %{y:.1%}<extra></extra>")
        fig.update_layout(xaxis_title="x", yaxis_title="Probability of 'yes'", yaxis=dict(range=[-0.02, 1.02], tickformat=".0%"))
        show(fig, 400)
        verdict = "yes" if p0 >= thr else "no"
        st.caption(f"At the current threshold ({thr:.2f}) this point is classed as **{verdict}**. "
                   "Change the threshold in the sidebar to slide the boundary.")

# ----------------------------------------------------------------------------- performance
with tab_perf:
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, zero_division=0)
    recall = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    cols = st.columns(5)
    for col, (name, value) in zip(cols, {"Accuracy": accuracy, "Precision": precision, "Recall": recall,
                                          "F1-score": f1, "ROC-AUC": auc}.items()):
        col.metric(name, f"{value:.3f}")
    callout(f"At threshold <b>{thr:.2f}</b> the model finds <b>{recall:.0%}</b> of the customers who really subscribe, "
            f"and <b>{precision:.0%}</b> of the customers it flags go on to subscribe.")

    r1a, r1b = st.columns(2)
    with r1a:
        cm = confusion_matrix(y_test, pred)
        pct = cm / cm.sum(axis=1, keepdims=True)
        text = [[f"{cm[i, j]:,}<br>{pct[i, j]:.0%} of row" for j in range(2)] for i in range(2)]
        fig = go.Figure(go.Heatmap(z=pct, x=["Predicted no", "Predicted yes"], y=["Actual no", "Actual yes"],
                                   text=text, texttemplate="%{text}", colorscale=[[0, "#FFFFFF"], [1, TEAL]],
                                   zmin=0, zmax=1, showscale=False, hoverinfo="skip"))
        fig.update_yaxes(autorange="reversed")
        show(fig, 340, "Confusion matrix")
    with r1b:
        fpr, tpr, _ = roc_curve(y_test, proba)
        tn, fp, fn, tp = cm.ravel()
        fig = go.Figure()
        fig.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color=TEAL, width=3), name=f"ROC (AUC {auc:.3f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=SLATE, dash="dash"), name="Random guess")
        fig.add_scatter(x=[fp / (fp + tn)], y=[tp / (tp + fn)], mode="markers",
                        marker=dict(size=13, color=AMBER, line=dict(color=INK, width=1.5)), name=f"Threshold {thr:.2f}")
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
        show(fig, 340, "ROC curve")

    r2a, r2b = st.columns(2)
    with r2a:
        grid_t = np.linspace(0.02, 0.98, 97)
        P, R, F = [], [], []
        for t in grid_t:
            p_t = (proba >= t).astype(int)
            P.append(precision_score(y_test, p_t, zero_division=0))
            R.append(recall_score(y_test, p_t, zero_division=0))
            F.append(f1_score(y_test, p_t, zero_division=0))
        fig = go.Figure()
        for name, vals, colour in (("Precision", P, TEAL), ("Recall", R, ROSE), ("F1-score", F, INK)):
            fig.add_scatter(x=grid_t, y=vals, mode="lines", name=name, line=dict(color=colour, width=2.5))
        fig.add_vline(x=thr, line_dash="dot", line_color=AMBER)
        fig.update_layout(xaxis_title="Threshold", yaxis_title="Score", yaxis_range=[0, 1.02])
        show(fig, 340, "Precision, recall and F1 across thresholds")
    with r2b:
        fig = go.Figure()
        for label, colour in ((0, SLATE), (1, TEAL)):
            fig.add_histogram(x=proba[y_test == label], name=f"Actually {'yes' if label else 'no'}",
                              marker_color=colour, opacity=0.65, xbins=dict(start=0, end=1, size=0.025),
                              histnorm="percent")
        fig.add_vline(x=thr, line_dash="dot", line_color=AMBER)
        fig.update_layout(barmode="overlay", xaxis_title="Predicted probability", yaxis_title="% of group")
        show(fig, 340, "Where the two groups score")

    st.markdown("#### Campaign targeting")
    st.write("If you can only call part of your list, calling the highest-scoring customers first finds more subscribers.")
    share = st.slider("Contact the top share of customers by score (%)", 1, 100, 20)
    order = np.argsort(-proba, kind="stable")
    cum = np.cumsum(y_test[order]) / y_test.sum()
    xs_g = np.arange(1, len(order) + 1) / len(order)
    idx = max(int(len(order) * share / 100) - 1, 0)
    captured = float(cum[idx])
    g1, g2, g3 = st.columns(3)
    g1.metric("Customers contacted", f"{int(len(order) * share / 100):,}")
    g2.metric("Subscribers reached", f"{captured:.0%}")
    g3.metric("Versus random calling", f"{captured / (share / 100):.1f}×")
    step = max(len(order) // 400, 1)
    fig = go.Figure()
    fig.add_scatter(x=xs_g[::step], y=cum[::step], mode="lines", line=dict(color=TEAL, width=3), name="Model ranking")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=SLATE, dash="dash"), name="Random order")
    fig.add_scatter(x=[share / 100], y=[captured], mode="markers",
                    marker=dict(size=13, color=AMBER, line=dict(color=INK, width=1.5)), name="Your selection")
    fig.update_layout(xaxis=dict(title="Share of customers contacted", tickformat=".0%"),
                      yaxis=dict(title="Share of subscribers reached", tickformat=".0%"))
    show(fig, 340, "Cumulative gains")

    with st.expander("Full classification report"):
        report = pd.DataFrame(classification_report(y_test, pred, target_names=["no", "yes"],
                                                    output_dict=True, zero_division=0)).T
        st.dataframe(report.round(3), width="stretch")

# ----------------------------------------------------------------------------- drivers
with tab_drivers:
    coefs = art["model"].named_steps["classifier"].coef_[0]
    table = pd.DataFrame({"Feature": [readable_feature(n, art) for n in art["names"]], "Log-odds": coefs})
    table["Odds ratio"] = np.exp(table["Log-odds"])
    top_up, top_down = table.sort_values("Log-odds").iloc[-1], table.sort_values("Log-odds").iloc[0]
    st.markdown("#### What pushes the prediction up or down")
    callout(f"Strongest push toward <b>yes</b>: <b>{top_up['Feature']}</b> (odds ×{top_up['Odds ratio']:.2f}). "
            f"Strongest push toward <b>no</b>: <b>{top_down['Feature']}</b> (odds ×{top_down['Odds ratio']:.2f}).")
    n_show = st.slider("Features to show", 5, min(30, len(table)), 15)
    shown = table.reindex(table["Log-odds"].abs().sort_values(ascending=False).index).head(n_show).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=shown["Log-odds"], y=shown["Feature"], orientation="h",
        marker_color=[TEAL if v > 0 else ROSE for v in shown["Log-odds"]], customdata=shown["Odds ratio"],
        hovertemplate="%{y}<br>log-odds %{x:+.2f}<br>odds ×%{customdata:.2f}<extra></extra>"))
    fig.update_layout(xaxis_title="Coefficient (log-odds). Right raises the chance of 'yes', left lowers it.")
    show(fig, max(320, 26 * n_show + 80))
    st.caption("Numeric inputs are standardized, so '+1 std dev' means one standard deviation above average. "
               "Categories are compared with customers who aren't in that category.")
    with st.expander("All coefficients"):
        st.dataframe(table.sort_values("Log-odds", ascending=False).round(3).reset_index(drop=True),
                     width="stretch", height=360)

# ----------------------------------------------------------------------------- predict
with tab_predict:
    st.markdown("#### Build a customer and see the score")
    st.caption("Start from a profile, then adjust any field. The result updates as you change it.")
    preset_cols = st.columns(len(PRESETS))
    for pc, name in zip(preset_cols, PRESETS):
        pc.button(name, key=f"preset_{name}", on_click=apply_preset, args=(name, profile), width="stretch")

    form_col, result_col = st.columns([1.1, 1], gap="large")
    opts = profile["options"]
    with form_col:
        with st.expander("Customer", expanded=True):
            a, b = st.columns(2)
            with a:
                st.slider(FIELD_LABELS["age"], 18, 95, key="p_age")
                st.selectbox(FIELD_LABELS["job"], opts["job"], key="p_job")
                st.selectbox(FIELD_LABELS["marital"], opts["marital"], key="p_marital")
                st.selectbox(FIELD_LABELS["education"], opts["education"], key="p_education")
            with b:
                st.number_input(FIELD_LABELS["balance"], min_value=-10000, max_value=120000, step=100, key="p_balance")
                st.selectbox(FIELD_LABELS["default"], opts["default"], key="p_default")
                st.selectbox(FIELD_LABELS["housing"], opts["housing"], key="p_housing")
                st.selectbox(FIELD_LABELS["loan"], opts["loan"], key="p_loan")
        with st.expander("Campaign contact", expanded=True):
            a, b = st.columns(2)
            with a:
                st.selectbox(FIELD_LABELS["contact"], opts["contact"], key="p_contact")
                st.selectbox(FIELD_LABELS["month"], opts["month"], key="p_month")
                st.slider(FIELD_LABELS["day"], 1, 31, key="p_day")
                if "duration" in art["feature_cols"]:
                    st.slider(FIELD_LABELS["duration"], 0, 3000, key="p_duration")
                else:
                    st.caption("Call duration is left out of this model.")
            with b:
                st.slider(FIELD_LABELS["campaign"], 1, 50, key="p_campaign")
                st.number_input(FIELD_LABELS["pdays"], min_value=-1, max_value=900, step=1, key="p_pdays")
                st.slider(FIELD_LABELS["previous"], 0, 40, key="p_previous")
                st.selectbox(FIELD_LABELS["poutcome"], opts["poutcome"], key="p_poutcome")

    values = {c: st.session_state[f"p_{c}"] for c in art["feature_cols"]}
    sample = pd.DataFrame([values])
    model = art["model"]
    probability = float(model.predict_proba(sample)[0, 1])
    likely = probability >= thr
    baseline_z, by_field = explain(art, sample)
    typical = float(sigmoid(baseline_z))

    with result_col:
        colour = TEAL if likely else SLATE
        label = f"Likely to subscribe at the {thr:.2f} threshold" if likely else f"Unlikely to subscribe at the {thr:.2f} threshold"
        st.markdown(f"""
        <div class="verdict">
          <div class="cap">Estimated chance of subscribing</div>
          <div class="num">{probability:.1%}</div>
          <div class="lab {'yes' if likely else 'no'}">{label}</div>
          <div class="meter"><div class="fill" style="width:{probability * 100:.1f}%;background:{colour}"></div>
            <div class="tick" style="left:calc({thr * 100:.0f}% - 1px)"></div></div>
          <div class="meter-scale"><span>0%</span><span>threshold {thr:.0%}</span><span>100%</span></div>
        </div>""", unsafe_allow_html=True)
        st.caption(f"The average customer profile scores {typical:.1%}."
                   + (" Balanced class weights push scores above the real-world subscription rate, so read them as a ranking, not a forecast."
                      if st.session_state["s_balanced"] else ""))

        top = sorted(by_field.items(), key=lambda kv: abs(kv[1]), reverse=True)[:8][::-1]
        fig = go.Figure(go.Bar(
            x=[v for _, v in top], y=[f"{FIELD_LABELS.get(k, k).split(' (')[0]}: {values[k]}" for k, _ in top],
            orientation="h", marker_color=[TEAL if v > 0 else ROSE for _, v in top],
            hovertemplate="%{y}<br>%{x:+.2f} log-odds<extra></extra>"))
        fig.update_layout(xaxis_title="Effect on the score compared with an average customer")
        show(fig, 330, "What moves this score")

    st.markdown("#### What if one thing changed?")
    wi_field = st.selectbox("Field to vary", art["feature_cols"], format_func=lambda c: FIELD_LABELS.get(c, c),
                            index=art["feature_cols"].index("age"))
    if wi_field in profile["numeric"]:
        lo, hi = profile["ranges"][wi_field]
        cur = values[wi_field]
        grid = np.unique(np.concatenate([np.linspace(min(lo, cur), max(hi, cur), 60), [cur]]))
        sweep = pd.concat([sample] * len(grid), ignore_index=True)
        sweep[wi_field] = grid
        curve = model.predict_proba(sweep)[:, 1]
        fig = go.Figure()
        fig.add_scatter(x=grid, y=curve, mode="lines", line=dict(color=TEAL, width=3), name="Probability")
        fig.add_hline(y=thr, line_dash="dot", line_color=AMBER, annotation_text=f"Threshold {thr:.2f}",
                      annotation_position="bottom right")
        fig.add_scatter(x=[cur], y=[probability], mode="markers", name="This customer",
                        marker=dict(size=13, color=INK, line=dict(color="#fff", width=2)))
        fig.update_layout(xaxis_title=FIELD_LABELS.get(wi_field, wi_field), yaxis_title="Probability",
                          yaxis=dict(tickformat=".0%", range=[0, max(1.0, float(curve.max()) * 1.05)]))
    else:
        cats = opts[wi_field]
        sweep = pd.concat([sample] * len(cats), ignore_index=True)
        sweep[wi_field] = cats
        curve = model.predict_proba(sweep)[:, 1]
        fig = go.Figure(go.Bar(
            x=cats, y=curve, marker_color=[INK if c == values[wi_field] else TEAL for c in cats],
            text=[f"{p:.0%}" for p in curve], textposition="outside",
            hovertemplate="%{x}<br>%{y:.1%}<extra></extra>"))
        fig.add_hline(y=thr, line_dash="dot", line_color=AMBER, annotation_text=f"Threshold {thr:.2f}",
                      annotation_position="top right")
        fig.update_layout(xaxis_title=FIELD_LABELS.get(wi_field, wi_field), yaxis_title="Probability",
                          yaxis=dict(tickformat=".0%", range=[0, min(1.15, max(float(curve.max()) * 1.2, thr * 1.2))]))
    show(fig, 340)
    st.caption("All other fields stay as set above. This is a demonstration model, not financial or marketing advice.")

# ----------------------------------------------------------------------------- concepts
with tab_concepts:
    st.markdown("#### Key concepts")
    st.latex(r"P(y=1\mid x)=\frac{1}{1+e^{-z}},\qquad z=\beta_0+\beta_1x_1+\cdots+\beta_px_p")
    concepts = {
        "Sigmoid": "Converts a score into a probability between 0 and 1.",
        "Threshold": "Turns a probability into a predicted class. Lowering it finds more subscribers but adds false alarms.",
        "Precision": "Of the customers predicted to subscribe, how many did?",
        "Recall": "Of the customers who subscribed, how many did the model find?",
        "F1-score": "The harmonic mean of precision and recall, a single number that punishes lopsided results.",
        "ROC-AUC": "How well the model ranks subscribers above non-subscribers across every threshold.",
        "Regularization (C)": "Keeps coefficients small to avoid overfitting. A smaller C means a stronger penalty.",
        "Scaling and encoding": "Numeric features are standardized and categories are one-hot encoded before fitting.",
        "Class weights": "Balanced weights make mistakes on the rare 'yes' class cost more during training.",
    }
    for name, text in concepts.items():
        with st.expander(name):
            st.write(text)
    callout("<b>About call duration.</b> It is known only after a call has happened. It's kept in by default for teaching, "
            "but a model that decides whom to call before a campaign should leave it out. Use the toggle in the sidebar to compare.",
            warn=True)
