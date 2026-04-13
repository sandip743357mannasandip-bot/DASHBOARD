import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

st.set_page_config(layout="wide")

# ================================
# DARK STYLE
# ================================
plt.style.use("dark_background")

# ================================
# TITLE
# ================================
st.title("⚽ Football Analytics PRO Dashboard")
st.markdown("### Full Model Analysis (xG • Score • Outcome • Probability)")

# ================================
# LOAD DATA
# ================================
AFTER_CSV = r"D:\FOOTBALL\RANDOM FOREST\AFTER PREDICTION.csv"
COMPLETE_CSV = r"D:\FOOTBALL\10 SEASON DATA OF LALIGA\football-project\COMPLTE PREDICTION.csv"

df_after = pd.read_csv(AFTER_CSV, encoding="latin1")
df_after.columns = df_after.columns.str.strip().str.lower()

df = pd.read_csv(COMPLETE_CSV, encoding="latin1")
df.columns = df.columns.str.strip()

# ================================
# PROCESS
# ================================
def parse_score(s):
    try:
        h, a = str(s).strip().split("-")
        return int(h), int(a)
    except:
        return 0, 0

df[["pred_h","pred_a"]] = pd.DataFrame(df["predict score"].apply(parse_score).tolist())
df[["act_h","act_a"]] = pd.DataFrame(df["actual score"].apply(parse_score).tolist())

df["pred_total"] = df["pred_h"] + df["pred_a"]
df["act_total"] = df["act_h"] + df["act_a"]
df["residual"] = df["act_total"] - df["pred_total"]

df["correct"] = df["predict outcome"] == df["actual outcome"]

# ================================
# KPIs
# ================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Matches", len(df))
col2.metric("Correct", df["correct"].sum())
col3.metric("Accuracy", f"{df['correct'].mean()*100:.1f}%")
col4.metric("Avg Goals", f"{df['act_total'].mean():.2f}")

# ================================
# TABS (MAIN POWER 🔥)
# ================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "xG Analysis",
    "Score Prediction",
    "Outcome Analysis",
    "Top-N Accuracy",
    "Probability Model"
])

# ================================
# TAB 1 — xG
# ================================
with tab1:

    xg_col = [c for c in df_after.columns if "xg" in c][0]
    goal_col = [c for c in df_after.columns if "goal" in c][0]

    st.subheader("xG vs Goals")

    fig, ax = plt.subplots()
    ax.scatter(df_after[xg_col], df_after[goal_col])
    ax.plot([0, df_after[xg_col].max()], [0, df_after[xg_col].max()])
    st.pyplot(fig)

    st.subheader("Residual")

    df_after["residual"] = df_after[goal_col] - df_after[xg_col]
    fig, ax = plt.subplots()
    ax.scatter(df_after[xg_col], df_after["residual"])
    ax.axhline(0)
    st.pyplot(fig)

# ================================
# TAB 2 — SCORE
# ================================
with tab2:

    st.subheader("Predicted vs Actual Score")

    fig, ax = plt.subplots()
    ax.scatter(df["pred_total"], df["act_total"])
    ax.plot([0,6],[0,6])
    st.pyplot(fig)

    st.subheader("Match Trends")

    fig, ax = plt.subplots()
    ax.plot(df["pred_total"], label="Predicted")
    ax.plot(df["act_total"], label="Actual")
    ax.legend()
    st.pyplot(fig)

# ================================
# TAB 3 — OUTCOME
# ================================
with tab3:

    st.subheader("Outcome Accuracy")

    correct = df["correct"].sum()
    wrong = len(df) - correct

    fig, ax = plt.subplots()
    ax.pie([correct, wrong], labels=["Correct","Wrong"], autopct="%1.1f%%")
    st.pyplot(fig)

    st.subheader("Confusion Matrix")

    matrix = pd.crosstab(df["predict outcome"], df["actual outcome"])
    fig, ax = plt.subplots()
    ax.imshow(matrix)
    st.pyplot(fig)

# ================================
# TAB 4 — TOP N
# ================================
with tab4:

    df["top5"] = df["1 FROM TOP 5"] == "YES"
    df["top3"] = df["1 FROM TOP 3"] == "YES"
    df["top2"] = df["1 FROM TOP 2"] == "YES"

    st.subheader("Top N Accuracy")

    st.write("Top 5:", df["top5"].mean()*100)
    st.write("Top 3:", df["top3"].mean()*100)
    st.write("Top 2:", df["top2"].mean()*100)

# ================================
# TAB 5 — PROBABILITY
# ================================
with tab5:

    st.subheader("Win Probability")

    fig, ax = plt.subplots()
    ax.plot(df["WIN PROBABILITY"], label="Win")
    ax.plot(df["DRAW PROBABILITY"], label="Draw")
    ax.plot(df["LOSS PROBABILITY"], label="Loss")
    ax.legend()

    st.pyplot(fig)
