# NeuroManifold: Sub-Band Multi-Manifold Riemannian SVM for Motor Imagery BCI

An end-to-end mathematical and machine learning framework for decoding 4-class human motor imagery EEG (**Left Hand, Right Hand, Both Feet, Tongue**) using **Symmetric Positive Definite (SPD) Riemannian Geometry**, **Multi-Band Tangent-Space Fusion**, **Unsupervised Session Alignment**, and **Support Vector Machines (SVMs)**.

This repository is structured to serve both as a reproducible codebase and as the **complete technical reference** for team members drafting the **IEEE-format research paper** and **presentation slides (PPT)**.

---

## 1. Problem Statement & Motivation

* **Real-World Goal:** Decode imagined body movements (`feet`, `left_hand`, `right_hand`, `tongue`) from non-invasive scalp EEG without physical movement, forming the core signal-decoding loop of a Brain-Computer Interface (BCI).
* **Why Standard Euclidean ML Struggles:** For each EEG trial, spatial co-variation across 22 scalp electrodes forms a $22 \times 22$ **Symmetric Positive Definite (SPD)** covariance matrix. SPD matrices lie on a **curved Riemannian manifold** $\mathcal{S}_{++}^{C}$ rather than a flat Euclidean space $\mathbb{R}^{d}$. Flattening them or measuring straight-line Euclidean distances distorts true geometric relationships, while cross-day EEG recordings introduce covariate shift between Session 1 (`0train`) and Session 2 (`1test`).
* **Our Approach:** We implement the core Riemannian manifold operations from scratch in NumPy, decompose motor imagery into four physiological sub-bands, align session manifolds to the Identity matrix, project onto flat tangent spaces, and classify using a regularized Linear SVM.

---

## 2. Dataset & Experimental Protocol

* **Dataset:** **BCI Competition IV – Dataset 2a** (`BNCI2014_001` via `MOABB`), recorded at Graz University of Technology.
* **Cohort & Signal Specs:**
  * **9 real human subjects** (`A01`–`A09` / `S01`–`S09`)
  * **22 scalp EEG channels** (`Fz`, `FC3`, `FC1`, `FCz`, `FC2`, `FC4`, `C5`, `C3`, `C1`, `Cz`, `C2`, `C4`, `C6`, `CP3`, `CP1`, `CPz`, `CP2`, `CP4`, `P1`, `Pz`, `P2`, `POz`) sampled at **250 Hz**
  * **4 motor imagery classes:** `feet`, `left_hand`, `right_hand`, `tongue` (72 trials per class per session)
  * **2 sessions per subject:** Session 1 (`0train`, 288 trials) and Session 2 (`1test`, 288 trials recorded on a separate day) — **5,184 total trials** across all 9 subjects.
* **Windowing & Leakage Prevention:** Trials are sliced to the active motor imagery window **`0.5s–3.5s`** (indices `125:875` at 250 Hz) to reduce cue-reaction and end-of-trial fatigue noise. All scaling, feature mapping, and SVM fitting use training data only; Day-2 session alignment uses unlabeled Day-2 covariance matrices only (zero test-label leakage).

---

## 3. Mathematical Formulation (For IEEE Paper Methods Section)

1. **Shrinkage-Regularized SPD Covariance:** Given a bandpass-filtered trial matrix $X \in \mathbb{R}^{C \times T}$ (mean-centered across time), sample covariance $\hat{\Sigma} = \frac{1}{T-1} X X^\top$ is regularized via shrinkage $\lambda = 0.03$:

   $$\Sigma_{\mathrm{reg}} = (1 - \lambda)\hat{\Sigma} + \lambda \frac{\mathrm{tr}(\hat{\Sigma})}{C} I_C$$

2. **Unsupervised Session Alignment:** Each session's SPD matrices are centered so their Riemannian geometric mean sits at the Identity matrix $I_C$:

   $$\tilde{\Sigma}_i = \bar{\Sigma}_{\mathrm{sess}}^{-1/2}\, \Sigma_i\, \bar{\Sigma}_{\mathrm{sess}}^{-1/2}$$

3. **Affine-Invariant Riemannian Mean & Tangent-Space Projection:** Using eigendecomposition $A = V \mathrm{diag}(\lambda_k) V^\top$ with $A^p = V \mathrm{diag}(\lambda_k^p) V^\top$, the Riemannian (Fréchet) mean $P$ is estimated via geodesic updates on $\mathcal{S}_{++}^{C}$, and each matrix is mapped to the tangent space at $P$:

   $$S_i = P^{1/2} \log\!\left(P^{-1/2} \tilde{\Sigma}_i P^{-1/2}\right) P^{1/2}$$

   Half-vectorizing the upper triangle of $S_i$ (scaling strict off-diagonals by $\sqrt{2}$) yields a **$253$-dimensional** feature vector per sub-band ($\frac{22 \times 23}{2} = 253$).

4. **4-Band Multi-Manifold Fusion & SVM:** Each trial is filtered into **Theta-Mu (4–12 Hz)**, **Mu-Low Beta (8–16 Hz)**, **Mid Beta (12–24 Hz)**, and **High Beta (16–32 Hz)**. Concatenating tangent vectors across all 4 bands produces a **$1,012$-dimensional** fused representation classified by a Linear SVM ($C = 0.05$) with `StandardScaler`.

---

## 4. 9-Subject Ablation Results (Mean Across S01–S09)

All four models use the same Linear SVM classifier (`C = 0.05`) so performance gains isolate the mathematical upgrades:

| Model ID | Model Name | Mathematical Upgrade | 5-Fold CV Acc | Day-2 Test Acc | Cohen's Kappa ($\kappa$) | Macro-F1 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **A1** | `A1_Euclidean_SVM` | Flattened $22\times 22$ covariances (253-D, flat space) | 49.77% | 52.12% | 0.3616 | 50.40% |
| **A2** | `A2_SingleBand_Riemann_SVM` | Broad 8–30 Hz Riemannian tangent space (253-D) | 54.29% | 54.36% | 0.3915 | 52.89% |
| **A3** | `A3_SubBand_Fusion_4Band` | 4-band multi-manifold tangent fusion (1,012-D) | 56.44% | 57.75% | 0.4367 | 56.84% |
| **A4** | `A4_Aligned_SubBand_Fusion_4Band` | 4-band fusion + unsupervised session alignment | **68.13%** | **67.52%** | **0.5669** | **67.46%** |

* **Key Takeaway:** Moving from Euclidean SVM (**A1**) to Aligned 4-Band Multi-Manifold Fusion (**A4**) improves mean Day-2 cross-session accuracy by **+15.40 percentage points** (`52.12%` → `67.52%`) and Cohen's Kappa by **+0.2053** (`0.3616` → `0.5669`) across all 9 subjects.
* **From-Scratch Verification:** Custom NumPy Riemannian mean and tangent-space code matches `pyRiemann` to **$7.51 \times 10^{-8}$** Frobenius norm difference.

---

## 5. Codebase Guide (Which File Does What)

| File | Purpose |
| :--- | :--- |
| `data_loader.py` | Downloads/loads `BNCI2014_001` via `MOABB` and splits `0train` (Session 1) and `1test` (Session 2). |
| `riemann_scratch.py` | From-scratch NumPy SPD shrinkage covariance, matrix powers/logs/exps via `eigh`, Fréchet mean, tangent-space projection, and `pyRiemann` verification. |
| `subband_pipeline.py` | Zero-phase Butterworth bandpass filter bank, single-band & multi-band tangent fusion, and single-subject diagnostic ablations. |
| `benchmark_all.py` | Runs `A1`–`A4` across all 9 subjects (`S01`–`S09`) and exports `benchmark_9_subjects.csv`. |
| `benchmark_9_subjects.csv` | Exact per-subject and per-model metrics (`cv_acc`, `test_acc`, `test_kappa`, `test_f1`) for tables and plots in the paper & PPT. |
| `inspect_dataset.py` | Utility script for inspecting dataset shapes and exporting sample CSV views. |
| `app.py` | Interactive Streamlit + 3D Plotly dashboard (`NeuroManifold`) with 9-subject ablation cards/charts, 4-step visual math walkthrough (including 3D SPD manifold & tangent sheet), and live Day-2 trial decoder. |

---

## 6. Blueprint for Teammates: IEEE Paper & Presentation (PPT)

### A. IEEE Conference Paper Outline (6–8 Pages, Two-Column IEEE Format)
1. **Title & Abstract:** State the 4-class motor imagery problem, SPD manifold curvature & cross-day drift, the A1–A4 ablation pipeline, and headline 9-subject results (`68.13%` CV, `67.52%` Day-2 test acc, `0.5669` Kappa).
2. **I. Introduction:** Motivation for non-invasive EEG BCIs, why Euclidean distance fails on SPD matrices, cross-session covariate shift, and summary of contributions.
3. **II. Related Work:** Classical EEG/CSP/SVM pipelines, Riemannian geometry for BCI (e.g., Barachant et al.), and filter-bank / multi-band tangent-space approaches.
4. **III. Mathematical Preliminaries & Proposed Method:**
   * SPD cone $\mathcal{S}_{++}^{C}$, affine-invariant metric, Fréchet mean, matrix logarithm via eigendecomposition, and half-vectorized tangent space.
   * Zero-phase sub-band decomposition (4 bands), unsupervised session alignment, and Linear SVM classification (`C = 0.05`).
   * Verification of scratch NumPy math against `pyRiemann` ($7.51 \times 10^{-8}$ Frobenius error).
5. **IV. Experimental Setup & Dataset:** BCI Competition IV-2a (`BNCI2014_001`), 9 subjects, 22 channels, 250 Hz, `0.5s–3.5s` window, Session 1 (`0train`) vs. Session 2 (`1test`), and evaluation metrics (Accuracy, Cohen's $\kappa$, Macro-F1).
6. **V. Results & Ablation Analysis:** Report the 9-subject mean table (`A1`–`A4`) from `benchmark_9_subjects.csv`, discuss per-subject variability (`S01`–`S09`), and analyze why sub-band fusion + session alignment outperforms flat and unaligned baselines.
7. **VI. Limitations & Future Work:** Offline session centering vs. online streaming adaptation, non-parametric significance testing (e.g., Wilcoxon tests) and per-class confusion analysis in the next phase, and cross-subject transfer.
8. **VII. Conclusion & References:** Concise wrap-up and key BCI/Riemannian citations.

### B. Presentation (PPT) Slide Plan (~10–12 Slides)
* **Slide 1 — Title Slide:** *NeuroManifold: Sub-Band Multi-Manifold Riemannian SVM for Motor Imagery BCI*, dataset (`BCI Comp IV-2a`), team & course info.
* **Slide 2 — The Problem:** Decoding 4 mental commands (`feet`, `left_hand`, `right_hand`, `tongue`) from noisy 22-channel EEG + why flat Euclidean geometry and Day 1 → Day 2 drift hurt accuracy.
* **Slide 3 — Dataset Overview:** BCI Competition IV-2a (`BNCI2014_001`), 9 human subjects, 22 channels @ 250 Hz, 288 train + 288 test trials/subject, active `0.5s–3.5s` window.
* **Slide 4 — From Brainwaves to SPD Matrices:** Raw EEG $\rightarrow$ Butterworth sub-bands $\rightarrow$ $22\times 22$ regularized covariance matrices (show screenshot/plot from `app.py` Tab 2).
* **Slide 5 — Riemannian Geometry & Tangent Space:** Curved manifold vs. flat tangent space, Fréchet mean, matrix log, and 3D manifold visual (`app.py` Tab 2).
* **Slide 6 — From-Scratch Math Verification:** NumPy implementation vs. `pyRiemann` ($7.51 \times 10^{-8}$ Frobenius difference) — proving mathematical rigor beyond library calls.
* **Slide 7 — Proposed Architecture (A4):** 4 sub-bands $\rightarrow$ SPD manifolds $\rightarrow$ unsupervised session alignment $\rightarrow$ tangent projection & fusion $\rightarrow$ Linear SVM (`C = 0.05`).
* **Slide 8 — Ablation Ladder (A1 → A4):** Progression cards & 9-subject summary (`52.12%` → `54.36%` → `57.75%` → `67.52%` test accuracy; `0.5669` Kappa).
* **Slide 9 — Per-Subject Results & Live Demo:** Per-subject chart from `benchmark_9_subjects.csv` + live demo of `app.py` (3D manifold & Day-2 trial decoder).
* **Slide 10 — Future Work & Conclusion:** Next-semester extensions (statistical significance tests, confusion matrix error analysis, online real-time alignment) and final summary.

---

## 7. Quickstart

```bash
pip install -r requirements.txt
python riemann_scratch.py
python benchmark_all.py
streamlit run app.py
