from pathlib import Path
try:  # keep first: importing sklearn/matplotlib before torch segfaults on macOS. Optional so the cloud deploy can skip torch.
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score,
                             precision_recall_fscore_support)

ROOT = Path(__file__).resolve().parent.parent  # repo root (this file lives in streamlit_app/)
RESULTS = ROOT / "results"
CKPTS = ROOT / "checkpoints"
MAX_LEN = 64

# Fine tuned models hosted on the Hugging Face Hub, shown in the "Try a model" tab: display name -> repo id.
# One repo can hold several models in sub folders: write "repo-id:sub-folder" (a plain "repo-id" uses the repo root).
# Each model folder needs config.json, model.safetensors, tokenizer.json and tokenizer_config.json.
HF_MODELS = {
    "Binary": "Tony281004/Technology-Design-Project:binary",
    "12 labels": "Tony281004/Technology-Design-Project:12labels",
}

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

tab_results, tab_llm, tab_live = st.tabs(["Results", "LLM vs BERT", "Try a model"])

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

# model -> (type, prediction file); binary sentiment only (the LLM experiments were run on this task)
LLM_RUNS = {
    "BERT B: Domain only": ("BERT (trained)", "pred_B1_bin.csv"),
    "BERT Sequential": ("BERT (trained)", "pred_Seq_bin.csv"),
    "BERT C: Mixed": ("BERT (trained)", "pred_C_bin.csv"),
    "BERT A: General only": ("BERT (trained)", "pred_A_bin.csv"),
    "Claude zero-shot": ("LLM (no training)", "pred_LLM_claude_zeroshot_binary.csv"),
    "Claude few-shot": ("LLM (no training)", "pred_LLM_claude_fewshot_binary.csv"),
    "Claude structured": ("LLM (no training)", "pred_LLM_claude_structured_binary.csv"),
    "Qwen-0.5B zero-shot": ("LLM (no training)", "pred_LLM_local_zeroshot_binary.csv"),
    "Qwen-0.5B few-shot": ("LLM (no training)", "pred_LLM_local_fewshot_binary.csv"),
    "Qwen-0.5B structured": ("LLM (no training)", "pred_LLM_local_structured_binary.csv"),
}

with tab_llm:
    st.caption("Binary sentiment (positive / negative) on the same 1,500 StockEmotions test rows for every model. "
               "LLMs are not trained: they only see a prompt. Saved results are shown; no LLM is called from this app.")
    llm_preds = {n: (t, load_preds(f)) for n, (t, f) in LLM_RUNS.items()}
    llm_preds = {n: (t, d) for n, (t, d) in llm_preds.items() if d is not None}
    if not any(t.startswith("LLM") for t, _ in llm_preds.values()):
        st.info("No LLM result files found in results/ (pred_LLM_*_binary.csv).")
    else:
        rows = []
        for n, (t, d) in llm_preds.items():
            p, r, f1, _ = precision_recall_fscore_support(d["true"], d["pred"], labels=["negative", "positive"],
                                                          average="macro", zero_division=0)
            rows.append({"model": n, "type": t, "accuracy": accuracy_score(d["true"], d["pred"]),
                         "precision": p, "recall": r, "macro_f1": f1, "invalid outputs": int((d["pred"] == "invalid").sum())})
        cmp = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).round(4)
        st.dataframe(cmp, hide_index=True, width='stretch')
        st.bar_chart(cmp.set_index("model")[["macro_f1"]])

        with st.expander("How the LLM prompts differ"):
            st.markdown(
                "- **zero-shot**: task and label names only.\n"
                "- **few-shot**: adds 2 examples per label, taken from the training split only.\n"
                "- **structured**: adds label definitions (positive = bullish, negative = bearish) and forces the answer to be one of the labels.\n"
                "- Claude is called through the API; Qwen2.5-0.5B-Instruct runs locally and picks the label with the highest log-probability.")

        pick2 = st.selectbox("Inspect a model", list(llm_preds), key="llm_pick")
        d2 = llm_preds[pick2][1]
        a, b = st.columns(2)
        rep2 = pd.DataFrame(classification_report(d2["true"], d2["pred"], zero_division=0, output_dict=True)).T.round(3)
        a.subheader("Per-class report")
        a.dataframe(rep2, width='stretch')
        b.subheader("Confusion matrix")
        labs = ["negative", "positive"] + (["invalid"] if (d2["pred"] == "invalid").any() else [])
        fig2, ax2 = plt.subplots(figsize=(5, 4.5))
        ConfusionMatrixDisplay.from_predictions(d2["true"], d2["pred"], labels=labs, xticks_rotation=45, cmap="Blues", ax=ax2, colorbar=False)
        plt.tight_layout()
        b.pyplot(fig2)

CKPT_LABELS = {"A_bin": "Binary", "S2a": "12 labels"}  # folder name -> display name
BASE_TOKENIZER = "cardiffnlp/twitter-roberta-base"


def find_models():
    """Display name -> where to load from: local checkpoints first, then Hugging Face Hub repos."""
    models = {}
    if HAS_TORCH:
        for p in sorted(CKPTS.glob("*/best/config.json")):
            folder = p.parent.parent.name
            models[CKPT_LABELS.get(folder, folder) + " (local)"] = str(p.parent)
        for label, repo in HF_MODELS.items():
            models[label + " (Hugging Face)"] = repo
    return models

def split_source(source):
    """A local folder is used as is; otherwise "repo-id" or "repo-id:sub-folder" on the Hugging Face Hub."""
    if Path(source).is_dir():
        return source, None
    repo, _, sub = source.partition(":")
    return repo, (sub or None)

@st.cache_resource(show_spinner="Loading model (first use can take a while)...")
def load_model(source):
    repo, sub = split_source(source)
    kw = {"subfolder": sub} if sub else {}
    try:
        tok = AutoTokenizer.from_pretrained(repo, **kw)  # a model folder that ships its tokenizer
    except Exception:
        tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER)
    model = AutoModelForSequenceClassification.from_pretrained(repo, low_cpu_mem_usage=True, **kw).eval()
    return tok, model


with tab_live:
    models = find_models()
    if not HAS_TORCH:
        st.info("Live prediction is disabled on this deployment (torch is not installed). "
                "The other tabs show the saved predictions of every strategy.")
    elif not models:
        st.info("No models found. Save a checkpoint in checkpoints/<name>/best (trainer.save_model(...)) "
                "or add a Hugging Face repo to HF_MODELS at the top of streamlit_app/app.py.")
    else:
        name = st.selectbox("Model", list(models))
        text = st.text_area("Text", "$TSLA earnings beat expectations, loading up more shares.")
        if st.button("Predict") and text.strip():
            try:
                tok, model = load_model(models[name])
            except Exception as e:
                st.error(f"Could not load this model: {type(e).__name__}: {e}")
                st.stop()
            enc = tok(text, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            with torch.no_grad():
                try:
                    probs = model(**enc).logits.float().softmax(-1)[0]
                except RuntimeError:  # some CPUs lack half precision kernels: fall back to float32
                    model.float()
                    probs = model(**enc).logits.softmax(-1)[0]
            out = pd.Series({model.config.id2label[i]: float(p) for i, p in enumerate(probs)})
            out = out.sort_values(ascending=False)
            st.success(f"Prediction: **{out.index[0]}** ({out.iloc[0]:.1%})")
            st.bar_chart(out)
        st.caption("Note: S2a / A_bin are general-only models (strategy A), so they transfer poorly to financial text.")
