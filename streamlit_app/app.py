"""Streamlit demo for the A/B/C emotion-classification experiments.

Run:  streamlit run app.py
Tab 1 shows saved results (results/*.csv). Tab 2 runs a live prediction with
any fine-tuned checkpoint found under checkpoints/<name>/best.
"""
from pathlib import Path

import torch  # keep first: importing sklearn/matplotlib before torch segfaults on macOS (OpenMP clash)
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score)

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
CKPTS = ROOT / "checkpoints"
MAX_LEN = 64

# prediction file -> display name, per task
RUNS = {
    "12-class emotion": {
        "A: General only": "pred_S2a.csv",
        "B: Domain only": "pred_S1.csv",
        "C: Mixed": "pred_S3.csv",
        "Sequential transfer": "pred_S2.csv",
    },
    "Binary sentiment": {
        "A: General only": "pred_A_bin.csv",
        "B: Domain only": "pred_B1_bin.csv",
        "C: Mixed": "pred_C_bin.csv",
        "Sequential transfer": "pred_Seq_bin.csv",
    },
}

st.set_page_config(page_title="Emotion Classification", layout="wide")
st.title("Emotion classification - Strategy (A/B/C)")
st.caption("GoEmotions (general) vs StockEmotions (finance), evaluated on the StockEmotions test set.")

tab_results, tab_live = st.tabs(["Results", "Try a model"])

@st.cache_data
def load_preds(file):
    path = RESULTS / file
    return pd.read_csv(path) if path.exists() else None

with tab_results:
    task = st.radio("Task", list(RUNS), horizontal=True)
    preds = {name: load_preds(f) for name, f in RUNS[task].items()}
    preds = {n: d for n, d in preds.items() if d is not None}
    if not preds:
        st.warning("No prediction files found in results/.")
        st.stop()

    summary = pd.DataFrame([
        {"strategy": n,
         "macro_f1": f1_score(d["true"], d["pred"], average="macro", zero_division=0),
         "accuracy": accuracy_score(d["true"], d["pred"])}
        for n, d in preds.items()]).round(4)

    left, right = st.columns([1, 1])
    left.dataframe(summary, hide_index=True, width='stretch')
    right.bar_chart(summary.set_index("strategy")[["macro_f1", "accuracy"]])

    pick = st.selectbox("Inspect a strategy", list(preds))
    d = preds[pick]
    c1, c2 = st.columns(2)
    report = pd.DataFrame(classification_report(d["true"], d["pred"], zero_division=0, output_dict=True)).T.round(3)
    c1.subheader("Per-class report")
    c1.dataframe(report, width='stretch')
    c2.subheader("Confusion matrix")
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay.from_predictions(d["true"], d["pred"], xticks_rotation=45, cmap="Blues", ax=ax, colorbar=False)
    plt.tight_layout()
    c2.pyplot(fig)

CKPT_LABELS = {"A_bin": "Binary", "S2a": "12 labels"}  # folder name -> display name


def find_checkpoints():
    return sorted(p.parent.parent.name for p in CKPTS.glob("*/best/config.json"))

@st.cache_resource
def load_model(name):
    path = CKPTS / name / "best"
    tok = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    return tok, model


with tab_live:
    names = find_checkpoints()
    if not names:
        st.info("No checkpoints found in checkpoints/<name>/best. Train a model in bert_final.ipynb and save it with trainer.save_model(...).")
    else:
        name = st.selectbox("Checkpoint", names, format_func=lambda n: CKPT_LABELS.get(n, n))
        text = st.text_area("Text", "$TSLA earnings beat expectations, loading up more shares.")
        if st.button("Predict") and text.strip():
            tok, model = load_model(name)
            enc = tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            with torch.no_grad():
                probs = model(**enc).logits.softmax(-1)[0]
            out = pd.Series({model.config.id2label[i]: float(p) for i, p in enumerate(probs)})
            out = out.sort_values(ascending=False)
            st.success(f"Prediction: **{out.index[0]}** ({out.iloc[0]:.1%})")
            st.bar_chart(out)
        st.caption("Note: S2a / A_bin are general-only models (strategy A), so they transfer poorly to financial text.")
