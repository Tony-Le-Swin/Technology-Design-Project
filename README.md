# Technology-Design-Project

Emotion Classification — Multi-Strategy BERT Fine-Tuning
Overview

This project compares three fine-tuning strategies for emotion classification, evaluated on a shared domain test set (StockEmotions, 12-label financial domain). All strategies use a common training pipeline and fixed hyperparameters so that results reflect the strategy, not the setup.

Datasets
GoEmotions — 28-label, general domain. Mapped to the 12-label StockEmotions label space for cross-dataset evaluation (neutral → ambiguous, full 12/12 coverage).
StockEmotions — 12-label, financial domain. Used as the common evaluation set for all three strategies.
Strategies
A — Domain-Only Fine-Tuning

Fine-tune the base model directly on StockEmotions (financial domain only). No exposure to GoEmotions. Serves as the domain-specific baseline.

B — Sequential Transfer Learning

Fine-tune first on GoEmotions (general domain), then continue fine-tuning the resulting checkpoint on StockEmotions (financial domain). Tests whether general-domain emotion knowledge transfers before domain specialization.

C — Mixed Training

Combine GoEmotions (label-mapped) and StockEmotions into a single training set and fine-tune once. Tests whether joint exposure to both domains outperforms sequential or domain-only training.

Model
Base model: roberta-base (final results)
Dev/debug model: distilroberta-base — used to validate the pipeline, label mapping, and seed control cheaply before running the full roberta-base sweep.
Evaluation
Metrics: Accuracy, F1 (macro), evaluated on the StockEmotions test set.
Each strategy is run over 2–3 seeds; results are averaged for the final report.
Reproducibility
Seeds are fixed across torch, numpy, random, and the DataLoader layer for every run.
All runs (across strategies, seeds, and models) are logged as entries in a single YAML config file — not one file per run.
Experiment tracking via Weights & Biases.
Downsampling

GoEmotions is downsampled to counteract neutral label dominance (~45% of the single-label data after mapping). Downsampling proportions are computed from the StockEmotions train split rather than hardcoded.