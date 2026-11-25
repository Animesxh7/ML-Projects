"""TreeWise: an interactive Decision Tree classification lab on UCI Red Wine Quality."""
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

st.set_page_config(page_title="TreeWise", page_icon="🌳", layout="wide")
DATA_PATH = Path(__file__).with_name("data") / "winequality-red.csv"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
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
[data-testid="stMetric"] {{ background: #05070B; border: 1px solid #303744; border-radius: 10px; padding: 14px 16px; }}
[data-testid="stMetricValue"] * {{ color: #F5F7FA !important; }}
[data-testid="stMetricLabel"] p {{ color: #5B6B80; font-size: .85rem; }}
button[data-baseweb="tab"] {{ font-weight: 600; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {GRID}; }}
.verdict {{ background: #05070B; border: 1px solid #303744; border-radius: 14px; padding: 22px 24px 18px; }}
.verdict .num {{ color: #F5F7FA; }}
.verdict .cap {{ color: #5B6B80; font-size: .9rem; }}
.verdict .num {{ font-family: 'Bricolage Grotesque', sans-serif; font-size: 4.2rem; line-height: 1.05; font-weight: 700; color: {INK}; }}
.verdict .lab {{ font-weight: 600; margin: .1rem 0 1rem; }}
.verdict .lab.yes {{ color: {TEAL}; }} .verdict .lab.no {{ color: {ROSE}; }}
.meter {{ position: relative; height: 12px; background: {GRID}; border-radius: 6px; }}
.meter .fill {{ position: absolute; inset: 0 auto 0 0; border-radius: 6px; }}
.meter .tick {{ position: absolute; top: -5px; width: 3px; height: 22px; background: {AMBER}; border-radius: 2px; }}
.meter-scale {{ display: flex; justify-content: space-between; color: #5B6B80; font-size: .8rem; margin-top: 8px; }}
.callout {{ background: #05070B; color: #E8ECF2; border: 1px solid #303744; border-left: 4px solid {TEAL}; border-radius: 8px; padding: 12px 16px; margin: .4rem 0 1rem; }}
.callout.warn {{ border-left-color: {AMBER}; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- helpers

def show(fig, height=340, title=None):
    """Apply the shared LogisticLens chart styling."""
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
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def download_data():
    DATA_PATH.parent.mkdir(parents=True,exist_ok=True)
    DATA_PATH.write_bytes(urlopen(DATA_URL,timeout=45).read())

@st.cache_data(show_spinner=False)
def load_data(path,mtime):
    return pd.read_csv(path,sep=";")

st.markdown("""<div class="hero"><h1>TreeWise</h1>
<p>Explore how a Decision Tree classifies red wine quality. Adjust the tree, inspect its rules,
and see how depth changes the model's performance.</p></div>""",unsafe_allow_html=True)

if not DATA_PATH.exists():
    st.info("The UCI Red Wine Quality dataset is not downloaded yet.")
    if st.button("Download dataset",type="primary"):
        try:
            with st.spinner("Getting the dataset from UCI..."):
                download_data()
            st.rerun()
        except Exception as exc:
            st.error(f"Download failed: {exc}. Run `python download_data.py` or place the CSV in data/.")
    st.stop()

df=load_data(str(DATA_PATH),DATA_PATH.stat().st_mtime)
TARGET="quality"
# Educational binary target: quality 6+ is considered higher quality.
df["quality_class"]=(df[TARGET]>=6).astype(int)
features=[c for c in df.columns if c not in [TARGET,"quality_class"]]
defaults=df[features].median()

with st.sidebar:
    st.markdown("### Tree settings")
    depth=st.slider("Maximum depth",1,15,4)
    leaf=st.slider("Minimum samples per leaf",1,30,5)
    split=st.slider("Minimum samples to split",2,50,10)
    criterion=st.selectbox("Split criterion",["gini","entropy","log_loss"])
    test_pct=st.slider("Test set size (%)",10,40,20,step=5)
    threshold=st.slider("Probability threshold for higher quality",0.10,0.90,0.50,step=0.05)
    st.caption("A deeper tree can fit more patterns but may overfit.")

X=df[features]; y=df["quality_class"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_pct/100,random_state=42,stratify=y)
model=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,min_samples_split=split,
                             criterion=criterion,class_weight="balanced",random_state=42)
model.fit(X_train,y_train)
prob=model.predict_proba(X_test)[:,1]
pred=(prob>=threshold).astype(int)
auc=roc_auc_score(y_test,prob)

st.markdown(f'<div class="status">Trained on {len(X_train):,} wines · tested on {len(X_test):,} · ROC-AUC {auc:.3f} · threshold {threshold:.2f}</div>',unsafe_allow_html=True)
tab_data,tab_perf,tab_tree,tab_predict,tab_ideas=st.tabs(["Explore data","Performance","Tree rules","Predict","Key concepts"])

with tab_data:
    a,b,c,d=st.columns(4)
    a.metric("Wine samples",f"{len(df):,}")
    b.metric("Chemical features",str(len(features)))
    c.metric("Higher-quality share",f"{y.mean():.1%}")
    d.metric("Quality range",f"{df.quality.min()}–{df.quality.max()}")
    selected=st.selectbox("Feature to explore",features)
    fig=go.Figure()
    for label,color in [(0,SLATE),(1,TEAL)]:
        fig.add_histogram(x=df.loc[y==label,selected],name="Quality < 6" if label==0 else "Quality ≥ 6",
                          marker_color=color,opacity=.65,histnorm="percent",nbinsx=35)
    fig.update_layout(barmode="overlay",xaxis_title=selected,yaxis_title="% within class")
    show(fig,370,f"Distribution of {selected}")
    grouped=df.groupby(TARGET).agg(samples=(TARGET,"size"),higher_rate=("quality_class","mean"))
    f=go.Figure(go.Bar(x=grouped.index,y=grouped.samples,marker_color=TEAL,
                       customdata=grouped.higher_rate,hovertemplate="Quality %{x}<br>%{y} samples<br>%{customdata:.1%} high class<extra></extra>"))
    f.update_layout(xaxis_title="Original quality score",yaxis_title="Number of wines")
    show(f,330,"Original quality score counts")
    with st.expander("Browse sample rows"):
        st.dataframe(df.sample(min(250,len(df)),random_state=42),use_container_width=True)

with tab_perf:
    acc=accuracy_score(y_test,pred); prec=precision_score(y_test,pred,zero_division=0)
    rec=recall_score(y_test,pred,zero_division=0); f1=f1_score(y_test,pred,zero_division=0)
    cols=st.columns(5)
    for col,(name,val) in zip(cols,{"Accuracy":acc,"Precision":prec,"Recall":rec,"F1-score":f1,"ROC-AUC":auc}.items()):
        col.metric(name,f"{val:.3f}")
    st.markdown('<div class="callout">Higher quality is defined here as the dataset score <b>6 or above</b>. This is a teaching label, not an expert wine rating.</div>',unsafe_allow_html=True)
    left,right=st.columns(2)
    with left:
        cm=confusion_matrix(y_test,pred)
        fig=go.Figure(go.Heatmap(z=cm,x=["Predicted lower","Predicted higher"],y=["Actual lower","Actual higher"],
                    text=cm,texttemplate="%{text}",colorscale=[[0,"#FFFFFF"],[1,TEAL]],showscale=False))
        fig.update_yaxes(autorange="reversed")
        show(fig,340,"Confusion matrix")
    with right:
        fpr,tpr,_=roc_curve(y_test,prob)
        fig=go.Figure()
        fig.add_scatter(x=fpr,y=tpr,mode="lines",name=f"ROC (AUC {auc:.3f})",line=dict(color=TEAL,width=3))
        fig.add_scatter(x=[0,1],y=[0,1],mode="lines",name="Random",line=dict(color=SLATE,dash="dash"))
        show(fig,340,"ROC curve")
    imp=pd.Series(model.feature_importances_,index=features).sort_values().tail(10)
    fig=go.Figure(go.Bar(x=imp.values,y=imp.index,orientation="h",marker_color=TEAL))
    fig.update_layout(xaxis_title="Feature importance")
    show(fig,350,"Most useful features in this fitted tree")
    with st.expander("Classification report"):
        st.dataframe(pd.DataFrame(classification_report(y_test,pred,target_names=["Lower quality","Higher quality"],output_dict=True,zero_division=0)).T.round(3),use_container_width=True)

with tab_tree:
    st.write("Each split asks a feature-based question. Follow the branches to reach a predicted class.")
    max_nodes=st.slider("Tree nodes to display",5,model.tree_.node_count,min(31,model.tree_.node_count),step=2)
    fig,ax=plt.subplots(figsize=(18,8))
    plot_tree(model,feature_names=features,class_names=["Lower (<6)","Higher (≥6)"],filled=True,
              rounded=True,impurity=False,proportion=True,max_depth=None,ax=ax)
    # Keep diagram legible by cropping to the first requested levels.
    ax.set_title("Decision Tree — fitted rules",fontsize=16)
    st.pyplot(fig,use_container_width=True)
    plt.close(fig)
    st.caption(f"Tree currently has {model.tree_.node_count} nodes and depth {model.get_depth()}. Adjust max depth in the sidebar.")

with tab_predict:
    st.markdown("#### Create a wine profile")
    st.caption("Move the feature values to estimate which quality group this model assigns.")
    preset=st.selectbox("Profile preset",["Dataset median","Higher alcohol / lower acidity","Lower alcohol / higher acidity"])
    values=defaults.copy()
    if preset=="Higher alcohol / lower acidity":
        values["alcohol"]=min(float(df.alcohol.max()),float(df.alcohol.quantile(.85)))
        values["volatile acidity"]=float(df["volatile acidity"].quantile(.2))
    elif preset=="Lower alcohol / higher acidity":
        values["alcohol"]=float(df.alcohol.quantile(.2))
        values["volatile acidity"]=float(df["volatile acidity"].quantile(.8))
    form,result=st.columns([1.15,1],gap="large")
    with form:
        with st.expander("Chemical measurements",expanded=True):
            for col in features:
                lo=float(df[col].quantile(.01)); hi=float(df[col].quantile(.99))
                values[col]=st.slider(col,min_value=lo,max_value=hi,value=float(np.clip(values[col],lo,hi)),step=(hi-lo)/100 if hi>lo else 0.01)
    sample=pd.DataFrame([values])
    p=float(model.predict_proba(sample)[0,1]); is_high=p>=threshold
    with result:
        st.metric("Estimated probability of higher-quality class",f"{p:.1%}")
        st.progress(int(np.clip(p,0,1)*100))
        st.success("Predicted higher-quality group (score ≥ 6)" if is_high else "Predicted lower-quality group (score < 6)")
        st.caption("This model is trained on chemical measurements and a simplified binary label.")
    st.markdown("#### What changes the prediction?")
    baseline=sample.copy()
    rows=[]
    for col in features:
        changed=sample.copy()
        changed[col]=float(df[col].median())
        rows.append((col,p-float(model.predict_proba(changed)[0,1])))
    effects=pd.DataFrame(rows,columns=["Feature","Score difference"]).sort_values("Score difference")
    fig=go.Figure(go.Bar(x=effects["Score difference"],y=effects.Feature,orientation="h",
                         marker_color=[TEAL if v>=0 else ROSE for v in effects["Score difference"]]))
    fig.update_layout(xaxis_title="Current score minus score with this feature reset to median")
    show(fig,430,"One-feature-at-a-time comparison")

with tab_ideas:
    st.markdown("#### Decision Tree in a nutshell")
    concepts={"Splitting":"The tree repeatedly divides samples using a feature and a cutoff.",
              "Gini / entropy":"Measures how mixed the classes are inside a node; a good split reduces impurity.",
              "Depth":"Limits how many decision levels the tree can learn. Smaller trees are easier to explain.",
              "Leaf":"A terminal node that outputs a class or class probability.",
              "Overfitting":"A very deep tree can memorize training patterns and generalize poorly.",
              "Feature importance":"Summarizes how much each feature helped reduce impurity across the fitted tree."}
    for name,desc in concepts.items():
        with st.expander(name): st.write(desc)
    st.markdown('<div class="callout">The target is derived from wine quality: <b>6–10 → higher quality</b>, <b>0–5 → lower quality</b>. Original scores remain available in Explore.</div>',unsafe_allow_html=True)
