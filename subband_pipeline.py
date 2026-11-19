import warnings
import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from data_loader import load_subject_raw_epochs
from riemann_scratch import (
    compute_spd_covariances_scratch,
    riemannian_mean_scratch,
    tangent_space_project_scratch,
)

warnings.filterwarnings("ignore")

SUB_BANDS_3 = {
    "Mu (8-12 Hz)": (8.0, 12.0),
    "Low Beta (13-20 Hz)": (13.0, 20.0),
    "High Beta (21-30 Hz)": (21.0, 30.0),
}

SUB_BANDS_2 = {
    "Mu (8-13 Hz)": (8.0, 13.0),
    "Beta (13-30 Hz)": (13.0, 30.0),
}


def bandpass_epochs(
    X: np.ndarray,
    fmin: float,
    fmax: float,
    fs: float = 250.0,
    order: int = 4,
) -> np.ndarray:
    """Applies a zero-phase Butterworth bandpass filter along the time axis."""
    nyquist = 0.5 * fs
    low = max(fmin / nyquist, 1e-4)
    high = min(fmax / nyquist, 0.9999)
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    filtered = sosfiltfilt(sos, X, axis=-1)
    return np.ascontiguousarray(filtered)


def extract_euclidean_cov_features(X: np.ndarray, shrinkage: float = 0.2) -> np.ndarray:
    """Baseline A1: Flattened upper-triangular covariance features without Riemannian geometry."""
    covs = compute_spd_covariances_scratch(X, shrinkage=shrinkage)
    n_trials, n_ch, _ = covs.shape
    triu_idx = np.triu_indices(n_ch)
    return np.array([covs[i][triu_idx] for i in range(n_trials)], dtype=np.float64)


def fit_transform_single_band_tangent(
    X_train: np.ndarray,
    X_test: np.ndarray,
    fmin: float,
    fmax: float,
    fs: float = 250.0,
    shrinkage: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fits Riemannian mean on X_train for one frequency band and projects
    both X_train and X_test onto that tangent space (zero test leakage).
    """
    X_tr_band = bandpass_epochs(X_train, fmin=fmin, fmax=fmax, fs=fs)
    X_te_band = bandpass_epochs(X_test, fmin=fmin, fmax=fmax, fs=fs)

    cov_tr = compute_spd_covariances_scratch(X_tr_band, shrinkage=shrinkage)
    cov_te = compute_spd_covariances_scratch(X_te_band, shrinkage=shrinkage)

    ref_mean = riemannian_mean_scratch(cov_tr, max_iter=35, tol=1e-6)
    ts_tr = tangent_space_project_scratch(cov_tr, ref_mean)
    ts_te = tangent_space_project_scratch(cov_te, ref_mean)
    return ts_tr, ts_te


def fit_transform_subband_fusion(
    X_train: np.ndarray,
    X_test: np.ndarray,
    bands: dict[str, tuple[float, float]] | None = None,
    fs: float = 250.0,
    shrinkage: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Builds separate SPD manifolds & tangent projections per sub-band,
    then concatenates them into a fused feature representation.
    """
    if bands is None:
        bands = SUB_BANDS_2

    train_parts = []
    test_parts = []
    used_bands = []

    for band_name, (fmin, fmax) in bands.items():
        ts_tr, ts_te = fit_transform_single_band_tangent(
            X_train, X_test, fmin=fmin, fmax=fmax, fs=fs, shrinkage=shrinkage
        )
        train_parts.append(ts_tr)
        test_parts.append(ts_te)
        used_bands.append(band_name)

    return np.hstack(train_parts), np.hstack(test_parts), used_bands


def _eval_pipeline(pipe, F_train, y_train, F_test, y_test, cv) -> dict[str, float]:
    cv_acc = float(np.mean(cross_val_score(pipe, F_train, y_train, cv=cv, scoring="accuracy")))
    pipe.fit(F_train, y_train)
    pred = pipe.predict(F_test)
    return {
        "cv_acc": cv_acc,
        "acc": float(accuracy_score(y_test, pred)),
        "kappa": float(cohen_kappa_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred, average="macro")),
    }


def run_subject_ablation(
    subject_id: int = 1,
    shrinkage: float = 0.2,
    C: float = 0.5,
    kernel: str = "linear",
    k_features: int | None = 120,
    bands: dict[str, tuple[float, float]] | None = None,
) -> dict[str, dict[str, float]]:
    X_train, y_train_str, X_test, y_test_str = load_subject_raw_epochs(
        subject_id=subject_id, fmin=4.0, fmax=38.0
    )

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_str)
    y_test = le.transform(y_test_str)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def build_clf(use_selector: bool = False):
        steps = [StandardScaler()]
        if use_selector and k_features is not None:
            steps.append(SelectKBest(score_func=f_classif, k=k_features))
        steps.append(SVC(C=C, kernel=kernel, random_state=42))
        return make_pipeline(*steps)

    # A1: Euclidean SVM
    F_eucl_tr = extract_euclidean_cov_features(X_train, shrinkage=shrinkage)
    F_eucl_te = extract_euclidean_cov_features(X_test, shrinkage=shrinkage)
    res_eucl = _eval_pipeline(build_clf(False), F_eucl_tr, y_train, F_eucl_te, y_test, cv)

    # A2: Single-Band Riemannian SVM (8-30 Hz)
    F_sb_tr, F_sb_te = fit_transform_single_band_tangent(
        X_train, X_test, fmin=8.0, fmax=30.0, shrinkage=shrinkage
    )
    res_sb = _eval_pipeline(build_clf(False), F_sb_tr, y_train, F_sb_te, y_test, cv)

    # A3a: Sub-Band Fusion (All features)
    F_fus_tr, F_fus_te, _ = fit_transform_subband_fusion(
        X_train, X_test, bands=bands or SUB_BANDS_2, shrinkage=shrinkage
    )
    res_fus_all = _eval_pipeline(build_clf(False), F_fus_tr, y_train, F_fus_te, y_test, cv)

    # A3b: Sub-Band Fusion + ANOVA Feature Selection (mitigates p > n overfitting)
    res_fus_sel = _eval_pipeline(build_clf(True), F_fus_tr, y_train, F_fus_te, y_test, cv)

    return {
        "A1_Euclidean_SVM": res_eucl,
        "A2_SingleBand_Riemann_SVM": res_sb,
        "A3a_SubBand_Fusion_All": res_fus_all,
        f"A3b_SubBand_Fusion_Top{k_features}": res_fus_sel,
    }


if __name__ == "__main__":
    print("Running Subject 1 Diagnostic Ablation (Linear kernel, shrinkage=0.2, Mu+Beta fusion)...")
    results = run_subject_ablation(
        subject_id=1,
        shrinkage=0.2,
        C=0.5,
        kernel="linear",
        k_features=120,
        bands=SUB_BANDS_2,
    )

    print("\nModel                          | 5-Fold CV Acc | Test Acc | Test Kappa | Test Macro-F1")
    print("-" * 86)
    for model_name, m in results.items():
        print(
            f"{model_name:<30} | "
            f"{m['cv_acc']*100:>11.2f}% | "
            f"{m['acc']*100:>7.2f}% | "
            f"{m['kappa']:>10.4f} | "
            f"{m['f1']*100:>12.2f}%"
        )