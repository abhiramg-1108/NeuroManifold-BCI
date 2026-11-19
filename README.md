# NeuroManifold: Sub-Band Multi-Manifold Riemannian SVM for Motor Imagery BCI

An end-to-end mathematical and machine learning framework for decoding 4-class human motor imagery EEG (**Left Hand, Right Hand, Both Feet, Tongue**) using **Symmetric Positive Definite (SPD) Riemannian Geometry**, **Multi-Band Tangent-Space Fusion**, and **Support Vector Machines**.

---

## Core Mathematical Pipeline

1. **Active Windowing & Sub-Band Filter Bank:** Slices 22-channel EEG trials (250 Hz) to the active motor imagery window (`0.5s–3.5s`) and decomposes signals via zero-phase Butterworth bandpass filters into four rhythms: **Theta-Mu (4–12 Hz)**, **Mu-Low Beta (8–16 Hz)**, **Mid Beta (12–24 Hz)**, and **High Beta (16–32 Hz)**.
2. **Regularized SPD Covariance Estimation:** Computes shrinkage-regularized $22 \times 22$ spatial covariance matrices per sub-band, embedding each trial onto a curved Riemannian manifold $\mathcal{M}_b$.
3. **Unsupervised Session Alignment:** Recenters each recording session's SPD matrices to the Identity matrix $I$ via their Riemannian geometric mean ($P^{-1/2} C P^{-1/2}$) to mitigate cross-day covariate shift without touching test labels.
4. **From-Scratch Riemannian Tangent-Space Projection:** Computes the Affine-Invariant Riemannian Fréchet Mean via geodesic gradient descent and projects matrices onto the flat tangent space $T_P\mathcal{M}_b$ via matrix logarithms (verified against `pyRiemann` to **$7.51 \times 10^{-8}$** Frobenius norm difference).
5. **Multi-Manifold Fusion & Linear SVM:** Concatenates half-vectorized tangent features across all four bands ($4 \times 253 = 1,012$ dimensions) and classifies mental commands using a regularized Linear SVM ($C = 0.05$).

---

## Dataset

* **Benchmark:** [BCI Competition IV – Dataset 2a](https://www.bbci.de/competition/iv/) (`BNCI2014_001` via `MOABB`)
* **Subjects & Protocol:** 9 real human subjects, 22 EEG channels at 250 Hz, 4 classes.
* **Evaluation:** 5-Fold Stratified Cross-Validation on **Session 1 (`0train`)** and blind cross-session evaluation on **Session 2 (`1test`, recorded on a separate day)** across all 9 subjects ($5,184$ total trials).

---

## 9-Subject Ablation Results (Mean Across S01–S09)

| Model | Description | 5-Fold CV Acc | Day-2 Test Acc | Cohen's Kappa ($\kappa$) | Macro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **A1. Euclidean SVM** | Flattened $22\times 22$ covariances (no manifold geometry) | 49.77% | 52.12% | 0.3616 | 50.40% |
| **A2. Single-Band Riemann SVM** | Broad 8–30 Hz Riemannian Tangent Space | 54.29% | 54.36% | 0.3915 | 52.89% |
| **A3. Sub-Band Fusion (4-Band)** | Multi-manifold tangent fusion across 4 frequency bands | 56.44% | 57.75% | 0.4367 | 56.84% |
| **A4. Aligned 4-Band Fusion (Ours)** | 4-band multi-manifold fusion + unsupervised session alignment | **68.13%** | **67.52%** | **0.5669** | **67.46%** |

---

## Repository Structure

```text
├── data_loader.py             # Loads & epochs BCI Competition IV-2a (BNCI2014_001)
├── riemann_scratch.py         # From-scratch SPD shrinkage, Fréchet mean & tangent-space math
├── subband_pipeline.py        # Filter-bank & single-subject diagnostic ablation pipeline
├── benchmark_all.py           # Full 9-subject cross-validation & Day-2 benchmark runner
├── benchmark_9_subjects.csv   # Saved per-subject & per-model evaluation metrics
├── app.py                     # Interactive Streamlit + 3D Plotly Riemannian BCI Lab
├── requirements.txt           # Python dependencies
└── README.md