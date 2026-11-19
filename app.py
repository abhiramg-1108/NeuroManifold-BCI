import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from benchmark_all import SUB_BANDS_4, fit_transform_aligned_subband_fusion
from data_loader import load_subject_raw_epochs
from riemann_scratch import (
    compute_spd_covariances_scratch,
    matrix_log_spd,
    matrix_power_spd,
    riemannian_mean_scratch,
    tangent_space_project_scratch,
)

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="NeuroManifold — Sub-Band Riemannian BCI Lab",
    page_icon="🧠",
    layout="wide",
)

CHANNELS_22 = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]


@st.cache_data
def load_benchmark_csv(path: str = "benchmark_9_subjects.csv") -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


@st.cache_resource
def load_subject1_walkthrough_data():
    """Loads Subject 1 (8-30 Hz, active 0.5-3.5s) for the visual math walkthrough."""
    X_tr, y_tr, _, _ = load_subject_raw_epochs(subject_id=1, fmin=8.0, fmax=30.0)
    X_w = X_tr[:, :, 125:875]
    covs_22 = compute_spd_covariances_scratch(X_w, shrinkage=0.03)
    return X_w, y_tr, covs_22


@st.cache_resource
def fit_subject_decoder(subject_id: int):
    X_tr, y_tr_s, X_te, y_te_s = load_subject_raw_epochs(subject_id=subject_id, fmin=4.0, fmax=38.0)
    X_tr_w = X_tr[:, :, 125:875]
    X_te_w = X_te[:, :, 125:875]

    le = LabelEncoder()
    y_tr = le.fit_transform(y_tr_s)
    y_te = le.transform(y_te_s)

    F_tr, F_te = fit_transform_aligned_subband_fusion(
        X_tr_w, X_te_w, bands=SUB_BANDS_4, shrinkage=0.03
    )
    clf = make_pipeline(
        StandardScaler(),
        SVC(C=0.05, kernel="linear", probability=True, random_state=42),
    )
    clf.fit(F_tr, y_tr)
    preds = clf.predict(F_te)
    probs = clf.predict_proba(F_te)
    return X_te, y_te_s, le.classes_, preds, probs


def _spd2_to_hyperboloid(mat: np.ndarray) -> tuple[float, float, float]:
    """Maps a 2x2 symmetric matrix [[a, b], [b, c]] to (v=(a-c)/2, b, u=(a+c)/2)."""
    a, b, c = mat[0, 0], mat[0, 1], mat[1, 1]
    return float(0.5 * (a - c)), float(b), float(0.5 * (a + c))


def build_manifold_to_tangent_3d(X_w: np.ndarray, y_tr: np.ndarray, n_per_class: int = 22) -> go.Figure:
    """
    Plots real Subject 1 (C3, C4) 2x2 SPD covariance matrices normalized to unit determinant
    on the curved hyperboloid manifold (u^2 - v^2 - b^2 = 1), their Riemannian Fréchet Mean P,
    the exact flat Tangent Space plane T_P M, and log-map projection lines from curved -> flat.
    """
    c3_i, c4_i = CHANNELS_22.index("C3"), CHANNELS_22.index("C4")
    lh_idx = np.where(y_tr == "left_hand")[0][:n_per_class]
    rh_idx = np.where(y_tr == "right_hand")[0][:n_per_class]
    sel_idx = np.concatenate([lh_idx, rh_idx])
    sel_labels = np.array(["left_hand"] * len(lh_idx) + ["right_hand"] * len(rh_idx))

    X_pair = X_w[sel_idx][:, [c3_i, c4_i], :]
    covs_2 = compute_spd_covariances_scratch(X_pair, shrinkage=0.03)

    # Normalize each 2x2 SPD matrix to unit determinant so all points sit cleanly on u = sqrt(1 + v^2 + b^2)
    covs_unit = np.array([C / np.sqrt(np.linalg.det(C)) for C in covs_2])

    # Compute exact Riemannian Fréchet Mean P (also has det(P) = 1)
    P = riemannian_mean_scratch(covs_unit, max_iter=35, tol=1e-7)
    P = P / np.sqrt(np.linalg.det(P))
    P_half = matrix_power_spd(P, 0.5)
    P_inv_half = matrix_power_spd(P, -0.5)
    v0, b0, u0 = _spd2_to_hyperboloid(P)

    # Project each curved point C_i onto the flat Tangent Space plane: P + S_i
    manifold_pts = np.array([_spd2_to_hyperboloid(C) for C in covs_unit])
    tangent_pts = []
    for C in covs_unit:
        S = P_half @ matrix_log_spd(P_inv_half @ C @ P_inv_half) @ P_half
        tangent_pts.append(_spd2_to_hyperboloid(P + S))
    tangent_pts = np.array(tangent_pts)

    # Build curved manifold mesh (hyperboloid bowl u = sqrt(1 + v^2 + b^2))
    span = max(np.max(np.abs(manifold_pts[:, :2])) * 1.25, 0.9)
    vg = np.linspace(-span, span, 30)
    bg = np.linspace(-span, span, 30)
    V_grid, B_grid = np.meshgrid(vg, bg)
    U_manifold = np.sqrt(1.0 + V_grid**2 + B_grid**2)

    # Exact flat Tangent Space plane touching P: u = (1 + v0*v + b0*b) / u0
    t_span = span * 0.85
    vt = np.linspace(v0 - t_span, v0 + t_span, 15)
    bt = np.linspace(b0 - t_span, b0 + t_span, 15)
    Vt_grid, Bt_grid = np.meshgrid(vt, bt)
    U_tangent = (1.0 + v0 * Vt_grid + b0 * Bt_grid) / u0

    fig = go.Figure()

    # 1. Curved Riemannian Manifold Bowl
    fig.add_trace(
        go.Surface(
            x=V_grid, y=B_grid, z=U_manifold,
            colorscale="Blues", opacity=0.32, showscale=False,
            name="Curved SPD Manifold (M)",
        )
    )

    # 2. Flat Tangent Space Sheet resting at Fréchet Mean P
    fig.add_trace(
        go.Surface(
            x=Vt_grid, y=Bt_grid, z=U_tangent,
            colorscale=[[0, "#FFB703"], [1, "#FB8500"]], opacity=0.28, showscale=False,
            name="Flat Tangent Space (T_P M)",
        )
    )

    # 3. Projection Lines (Curved Point -> Flat Tangent Point)
    line_x, line_y, line_z = [], [], []
    for m_pt, t_pt in zip(manifold_pts, tangent_pts):
        line_x.extend([m_pt[0], t_pt[0], None])
        line_y.extend([m_pt[1], t_pt[1], None])
        line_z.extend([m_pt[2], t_pt[2], None])
    fig.add_trace(
        go.Scatter3d(
            x=line_x, y=line_y, z=line_z,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.45)", width=2, dash="dot"),
            name="Log-Map Projection (Curved → Flat)",
        )
    )

    # 4. Real Trials on Curved Manifold & Projected onto Flat Tangent Plane
    colors = {"left_hand": "#00E5A8", "right_hand": "#FF4D6D"}
    for cls_name, col in colors.items():
        mask = sel_labels == cls_name
        fig.add_trace(
            go.Scatter3d(
                x=manifold_pts[mask, 0], y=manifold_pts[mask, 1], z=manifold_pts[mask, 2],
                mode="markers",
                marker=dict(size=5, color=col, symbol="circle"),
                name=f"{cls_name} (On Curved Manifold)",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=tangent_pts[mask, 0], y=tangent_pts[mask, 1], z=tangent_pts[mask, 2],
                mode="markers",
                marker=dict(size=5, color=col, symbol="diamond", line=dict(color="white", width=1)),
                name=f"{cls_name} (Flattened on Tangent Sheet)",
            )
        )

    # 5. Riemannian Fréchet Mean P
    fig.add_trace(
        go.Scatter3d(
            x=[v0], y=[b0], z=[u0],
            mode="markers",
            marker=dict(size=9, color="#FFD166", symbol="x"),
            name="Riemannian Fréchet Mean (P)",
        )
    )

    fig.update_layout(
        title="Real Subject 1 (C3 vs C4): Curved SPD Manifold → Log-Map → Flat Tangent Space",
        scene=dict(
            xaxis_title="v = (Var C3 − Var C4)/2",
            yaxis_title="b = Cov(C3, C4)",
            zaxis_title="u = (Var C3 + Var C4)/2",
        ),
        height=540,
        margin=dict(l=0, r=0, t=42, b=0),
        legend=dict(font=dict(size=10)),
    )
    return fig


def build_svm_boundary_comparison(covs_22: np.ndarray, y_tr: np.ndarray) -> tuple[go.Figure, go.Figure]:
    """
    Compares 2D PCA of raw Euclidean covariances vs. 2D PCA of Riemannian Tangent Space
    for Left Hand vs Right Hand trials, drawing the actual Linear SVM separating hyperplane.
    """
    mask = np.isin(y_tr, ["left_hand", "right_hand"])
    covs_bin = covs_22[mask]
    y_bin = (y_tr[mask] == "right_hand").astype(int)

    # Euclidean features
    triu = np.triu_indices(covs_bin.shape[1])
    F_eucl = np.array([C[triu] for C in covs_bin])

    # Riemannian Tangent Space features
    ref_mean = riemannian_mean_scratch(covs_bin, max_iter=25, tol=1e-6)
    F_tang = tangent_space_project_scratch(covs_bin, ref_mean)

    def _make_2d_svm_plot(F: np.ndarray, title: str) -> go.Figure:
        F_scaled = StandardScaler().fit_transform(F)
        pts_2d = PCA(n_components=2, random_state=42).fit_transform(F_scaled)
        svm2d = SVC(C=0.5, kernel="linear").fit(pts_2d, y_bin)
        acc_2d = svm2d.score(pts_2d, y_bin)

        x_min, x_max = pts_2d[:, 0].min() - 1.0, pts_2d[:, 0].max() + 1.0
        y_min, y_max = pts_2d[:, 1].min() - 1.0, pts_2d[:, 1].max() + 1.0
        xx = np.linspace(x_min, x_max, 120)
        w = svm2d.coef_[0]
        b = svm2d.intercept_[0]
        yy = -(w[0] * xx + b) / (w[1] + 1e-12)

        fig = go.Figure()
        for cls_id, name, col in [(0, "Left Hand", "#00E5A8"), (1, "Right Hand", "#FF4D6D")]:
            m = y_bin == cls_id
            fig.add_trace(
                go.Scatter(
                    x=pts_2d[m, 0], y=pts_2d[m, 1],
                    mode="markers", name=name,
                    marker=dict(size=8, color=col, opacity=0.85),
                )
            )
        fig.add_trace(
            go.Scatter(
                x=xx, y=yy, mode="lines",
                name="Linear SVM Hyperplane",
                line=dict(color="#FFD166", width=3, dash="dash"),
            )
        )
        fig.update_layout(
            title=f"{title} (2D Linear SVM Acc: {acc_2d*100:.1f}%)",
            xaxis_title="PC 1",
            yaxis_title="PC 2",
            yaxis=dict(range=[y_min, y_max]),
            height=360,
            margin=dict(l=10, r=10, t=42, b=10),
        )
        return fig

    return (
        _make_2d_svm_plot(F_eucl, "Euclidean Space (No Manifold Math)"),
        _make_2d_svm_plot(F_tang, "Riemannian Tangent Space (Flattened)"),
    )


st.title("NeuroManifold: Sub-Band Multi-Manifold Riemannian SVM for BCI")
st.caption(
    "Real human motor imagery EEG (BCI Competition IV – Dataset 2a, BNCI2014_001) • "
    "From-scratch SPD & Fréchet mean math + 4-band tangent-space fusion + session alignment"
)

tab1, tab2, tab3 = st.tabs(
    [
        "1. 9-Subject Benchmark & Ablations",
        "2. Visual Math Walkthrough (EEG → Matrix → 3D Manifold → SVM)",
        "3. Live Day-2 Trial Decoder",
    ]
)

with tab1:
    df_bench = load_benchmark_csv()
    if df_bench is None:
        st.warning("Run `python benchmark_all.py` first to generate `benchmark_9_subjects.csv`.")
    else:
        summary = (
            df_bench.groupby("model")[["cv_acc", "test_acc", "test_kappa", "test_f1"]]
            .mean()
            .reset_index()
        )
        acc_lookup = dict(zip(summary["model"], summary["test_acc"] * 100))

        st.subheader("How Our 4 Models Progress (1 Math Upgrade per Step)")
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            with st.container(border=True):
                st.markdown("**A1. Euclidean SVM**")
                st.caption("Baseline (Flat Space)")
                st.metric("Day-2 Test Acc", f"{acc_lookup.get('A1_Euclidean_SVM', 0):.2f}%")
                st.markdown(
                    "Flattens raw $22\\times 22$ covariances (253 dims) straight into Linear SVM. "
                    "**Ignores curved SPD geometry.**"
                )

        with m2:
            with st.container(border=True):
                st.markdown("**A2. Single-Band Riemann**")
                st.caption("+ Curved Manifold Math")
                delta2 = acc_lookup.get("A2_SingleBand_Riemann_SVM", 0) - acc_lookup.get("A1_Euclidean_SVM", 0)
                st.metric("Day-2 Test Acc", f"{acc_lookup.get('A2_SingleBand_Riemann_SVM', 0):.2f}%", f"{delta2:+.2f}%")
                st.markdown(
                    "Maps broad **8–30 Hz** SPD matrices onto flat **Tangent Space** via Fréchet mean & matrix log (253 dims)."
                )

        with m3:
            with st.container(border=True):
                st.markdown("**A3. 4-Band Fusion**")
                st.caption("+ Multi-Manifold Split")
                delta3 = acc_lookup.get("A3_SubBand_Fusion_4Band", 0) - acc_lookup.get("A2_SingleBand_Riemann_SVM", 0)
                st.metric("Day-2 Test Acc", f"{acc_lookup.get('A3_SubBand_Fusion_4Band', 0):.2f}%", f"{delta3:+.2f}%")
                st.markdown(
                    "Splits EEG into **4 motor bands** (4–12, 8–16, 12–24, 16–32 Hz) and fuses 4 separate manifolds (1,012 dims)."
                )

        with m4:
            with st.container(border=True):
                st.markdown("**A4. Aligned 4-Band (Ours)**")
                st.caption("+ Session Alignment")
                delta4 = acc_lookup.get("A4_Aligned_SubBand_Fusion_4Band", 0) - acc_lookup.get("A3_SubBand_Fusion_4Band", 0)
                st.metric("Day-2 Test Acc", f"{acc_lookup.get('A4_Aligned_SubBand_Fusion_4Band', 0):.2f}%", f"{delta4:+.2f}%")
                st.markdown(
                    "Recenters each day's manifold to **Identity $I$** (unsupervised) to remove Day 1 → Day 2 brainwave drift."
                )

        st.divider()
        st.subheader("Full 9-Subject Benchmark Table (BCI Competition IV – 2a)")
        display_df = summary.copy()
        for col in ["cv_acc", "test_acc", "test_f1"]:
            display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"
        display_df["test_kappa"] = display_df["test_kappa"].round(4)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.subheader("Per-Subject Day-2 Test Accuracy (%)")
        pivot_acc = df_bench.pivot(index="subject", columns="model", values="test_acc") * 100
        st.bar_chart(pivot_acc, height=340)

with tab2:
    X_w, y_tr_s1, covs_22 = load_subject1_walkthrough_data()

    st.subheader("Step 1 & Step 2: Raw EEG Brainwaves → 22×22 SPD Covariance Matrix")
    sample_idx = st.slider("Inspect Subject 1 Training Trial", min_value=0, max_value=len(X_w) - 1, value=2)
    col_a, col_b = st.columns([1.15, 1])

    with col_a:
        t_sec = np.arange(X_w.shape[2]) / 250.0 + 0.5
        fig_wave = go.Figure()
        for ch_name, col in [("C3", "#00E5A8"), ("Cz", "#FFD166"), ("C4", "#FF4D6D")]:
            ci = CHANNELS_22.index(ch_name)
            fig_wave.add_trace(
                go.Scatter(x=t_sec, y=X_w[sample_idx, ci], mode="lines", name=ch_name, line=dict(color=col))
            )
        fig_wave.update_layout(
            title=f"Step 1: Raw 8–30 Hz EEG (Trial {sample_idx} — '{y_tr_s1[sample_idx]}')",
            xaxis_title="Active Window (s)",
            yaxis_title="Amplitude (μV)",
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_wave, use_container_width=True)

    with col_b:
        fig_cov = go.Figure(
            data=go.Heatmap(
                z=covs_22[sample_idx],
                x=CHANNELS_22,
                y=CHANNELS_22,
                colorscale="Viridis",
            )
        )
        fig_cov.update_layout(
            title="Step 2: 22×22 SPD Spatial Covariance Matrix",
            height=360,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_cov, use_container_width=True)

    st.divider()
    st.subheader("Step 3: Curved Riemannian Manifold → Log-Map → Flat Tangent Space (3D)")
    st.markdown(
        "Each **circle** is a real Subject 1 `(C3, C4)` covariance matrix sitting on the curved blue SPD manifold. "
        "We compute their geometric center (**gold X: Fréchet Mean $P$**), place the **flat orange Tangent Space sheet** "
        "right on $P$, and project every curved point onto the flat sheet (**diamonds**) using the matrix logarithm."
    )
    st.plotly_chart(build_manifold_to_tangent_3d(X_w, y_tr_s1), use_container_width=True)

    st.divider()
    st.subheader("Step 4: Why Flattening Works — Linear SVM Decision Boundary")
    fig_euc, fig_tan = build_svm_boundary_comparison(covs_22, y_tr_s1)
    c_left, c_right = st.columns(2)
    with c_left:
        st.plotly_chart(fig_euc, use_container_width=True)
    with c_right:
        st.plotly_chart(fig_tan, use_container_width=True)

with tab3:
    col_l, col_r = st.columns([1, 2])
    with col_l:
        sid = st.selectbox("Select Subject", options=list(range(1, 10)), index=0)
        trial_idx = st.slider("Unseen Day-2 Test Trial", min_value=0, max_value=287, value=0)

    X_te, y_te_s, classes, preds, probs = fit_subject_decoder(sid)
    true_label = y_te_s[trial_idx]
    pred_label = classes[preds[trial_idx]]
    is_correct = true_label == pred_label

    with col_l:
        st.markdown(f"**Ground Truth:** `{true_label}`")
        st.markdown(f"**Model Prediction:** `{pred_label}`")
        if is_correct:
            st.success("Correctly decoded on unseen Day-2 session!")
        else:
            st.error("Misclassified on this trial (cross-day variance).")

        prob_df = pd.DataFrame({"Class": classes, "SVM Probability": probs[trial_idx]})
        st.bar_chart(prob_df.set_index("Class"), height=220)

    with col_r:
        st.subheader(f"Trial {trial_idx} — Key Motor Channels (C3, Cz, C4)")
        t_axis = np.arange(X_te.shape[2]) / 250.0
        fig_eeg = go.Figure()
        for ch_name in ["C3", "Cz", "C4"]:
            ch_i = CHANNELS_22.index(ch_name)
            fig_eeg.add_trace(
                go.Scatter(x=t_axis, y=X_te[trial_idx, ch_i], mode="lines", name=ch_name)
            )
        fig_eeg.update_layout(
            xaxis_title="Time (seconds)",
            yaxis_title="Amplitude (μV)",
            height=340,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig_eeg, use_container_width=True)