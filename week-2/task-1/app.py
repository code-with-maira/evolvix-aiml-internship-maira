import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare",
            "Sex_male", "Embarked_Q", "Embarked_S"]

st.set_page_config(page_title="Titanic Model Comparison", page_icon="🚢", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(URL)
    data = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].copy()
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])
    data["Sex_male"] = (data["Sex"] == "male").astype(int)
    data["Embarked_Q"] = (data["Embarked"] == "Q").astype(int)
    data["Embarked_S"] = (data["Embarked"] == "S").astype(int)
    return data[["Survived"] + FEATURES]


@st.cache_resource
def train_models():
    data = load_data()
    X, y = data[FEATURES], data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42))]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(max_depth=4, random_state=42))]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42))]),
    }

    rows, roc, cms = [], {}, {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob)
        rows.append({"Model": name,
                     "Accuracy": accuracy_score(y_test, pred),
                     "Precision": precision_score(y_test, pred),
                     "Recall": recall_score(y_test, pred),
                     "F1-score": f1_score(y_test, pred),
                     "ROC-AUC": auc})
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc[name] = (fpr, tpr, auc)
        cms[name] = confusion_matrix(y_test, pred)

    results = pd.DataFrame(rows).set_index("Model").round(3)
    return models, results, roc, cms


models, results, roc, cms = train_models()

st.title("🚢 Titanic Survival: Model Comparison")
st.caption("Logistic Regression vs Decision Tree vs Random Forest | Week 2, Task 1")

tab_predict, tab_compare, tab_about = st.tabs(["🔮 Predict", "📊 Compare models", "ℹ️ About"])

# ---------------------------------------------------------------- Predict
with tab_predict:
    st.subheader("Enter passenger details")
    c1, c2, c3 = st.columns(3)
    with c1:
        pclass = st.selectbox("Ticket class", [1, 2, 3], index=2,
                              help="1 = highest, 3 = lowest")
        sex = st.radio("Sex", ["female", "male"], horizontal=True)
    with c2:
        age = st.slider("Age", 0, 80, 28)
        fare = st.slider("Fare paid", 0.0, 520.0, 15.0, step=0.5)
    with c3:
        sibsp = st.number_input("Siblings / spouses aboard", 0, 8, 0)
        parch = st.number_input("Parents / children aboard", 0, 6, 0)
        embarked = st.selectbox("Port of embarkation",
                                ["S (Southampton)", "C (Cherbourg)", "Q (Queenstown)"])

    port = embarked[0]
    row = pd.DataFrame([{
        "Pclass": pclass, "Age": age, "SibSp": sibsp, "Parch": parch, "Fare": fare,
        "Sex_male": int(sex == "male"),
        "Embarked_Q": int(port == "Q"), "Embarked_S": int(port == "S"),
    }])[FEATURES]

    st.divider()
    st.subheader("Survival probability by model")
    cols = st.columns(3)
    probs = {}
    for col, (name, pipe) in zip(cols, models.items()):
        p = float(pipe.predict_proba(row)[0, 1])
        probs[name] = p
        with col:
            st.metric(name, f"{p * 100:.1f}%")
            st.progress(p)
            st.write("✅ Likely survived" if p >= 0.5 else "❌ Likely did not survive")

    agree = sum(p >= 0.5 for p in probs.values())
    if agree in (0, 3):
        st.info("All three models agree on this passenger.")
    else:
        st.warning("The models disagree, so treat this prediction with caution.")

# ---------------------------------------------------------------- Compare
with tab_compare:
    st.subheader("Test-set results")
    st.dataframe(results.style.highlight_max(axis=0, color="#c8f7c5").format("{:.3f}"),
                 width="stretch")
    best = results["ROC-AUC"].idxmax()
    st.success(f"Highest ROC-AUC: **{best}** ({results.loc[best, 'ROC-AUC']:.3f})")

    left, right = st.columns(2)
    with left:
        st.subheader("ROC curves")
        fig, ax = plt.subplots(figsize=(6, 5))
        for name, (fpr, tpr, auc) in roc.items():
            ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="Random guess")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    with right:
        st.subheader("Confusion matrix")
        choice = st.selectbox("Model", list(cms.keys()))
        fig2, ax2 = plt.subplots(figsize=(5, 4.4))
        ConfusionMatrixDisplay(cms[choice], display_labels=["Died", "Survived"]).plot(
            ax=ax2, cmap="Blues", colorbar=False)
        st.pyplot(fig2)

# ---------------------------------------------------------------- About
with tab_about:
    st.markdown("""
**Data:** Titanic dataset (891 passengers), 80/20 stratified train/test split.

**Models**
- Logistic Regression (median imputation + scaling)
- Decision Tree (max depth 4)
- Random Forest (200 trees, max depth 6)

**Metrics:** accuracy, precision, recall, F1-score and ROC-AUC, all measured on the
20% test set the models never saw during training.

*Predictions are statistical estimates based on a small historical dataset,
not certainties.*
""")
