import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from data_loader import load_subject_raw_epochs
from riemann_scratch import (
    compute_spd_covariances_scratch,
    matrix_power_spd,
    riemannian_mean_scratch,
    tangent_space_project_scratch,
)
from subband_pipeline import bandpass_epochs, extract_euclidean_cov_features

warnings.filterwarnings("ignore")

SUB_BANDS_4 = {
    "Theta-Mu (4-12 Hz)": (4.0, 12.0),
    "Mu-LowBeta (8-16 Hz)": (8.0, 16.0),
    "MidBeta (12-24 Hz)": (12.0, 24.0),
    "HighBeta (16-32 Hz)": (16.0, 32.0),
}


def recenter_spd_to_identity(covs: np.ndarray, ref_mean: np.ndarray) -> np.ndarray:
    """
    Unsupervised session alignment on the SPD manifold:
    maps covariance matrices so their Riemannian geometric mean sits at Identity I,
    reducing cross-session covariate shift without using labels.
    """
    inv_sqrt = matrix_power_spd(ref_mean, -0.5)
    aligned = np.empty_like(covs)
    for i, C in enumerate(covs):
        mat = inv_sqrt @ C @ inv_sqrt
        aligned[i] = 0.5 * (mat + mat.T)
    return aligned


def fit_transform_aligned_subband_fusion(
    X_train: np.ndarray,
    X_test: np.ndarray,
    bands: dict[str, tuple[float, float]] = SUB_BANDS_4,
    fs: float = 250.0,
    shrinkage: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sub-band multi-manifold fusion with session-wise SPD recentering
    and tangent-space projection at the training manifold mean.
    """
    train_blocks, test_blocks = [], []

    for _, (fmin, fmax) in bands.items():
        X_tr_b = bandpass_epochs(X_train, fmin=fmin, fmax=fmax, fs=fs)
        X_te_b = bandpass_epochs(X_test, fmin=fmin, fmax=fmax, fs=fs)

        cov_tr = compute_spd_covariances_scratch(X_tr_b, shrinkage=shrinkage)
        cov_te = compute_spd_covariances_scratch(X_te_b, shrinkage=shrinkage)

        mean_tr = riemannian_mean_scratch(cov_tr, max_iter=30, tol=1e-6)
        mean_te = riemannian_mean_scratch(cov_te, max_iter=30, tol=1e-6)

        cov_tr_al = recenter_spd_to_identity(cov_tr, mean_tr)
        cov_te_al = recenter_spd_to_identity(cov_te, mean_te)

        ref_mean = riemannian_mean_scratch(cov_tr_al, max_iter=25, tol=1e-6)
        train_blocks.append(tangent_space_project_scratch(cov_tr_al, ref_mean))
        test_blocks.append(tangent_space_project_scratch(cov_te_al, ref_mean))

    return np.hstack(train_blocks), np.hstack(test_blocks)


def fit_transform_unaligned_subband_fusion(
    X_train: np.ndarray,
    X_test: np.ndarray,
    bands: dict[str, tuple[float, float]] = SUB_BANDS_4,
    fs: float = 250.0,
    shrinkage: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sub-band multi-manifold fusion without session recentering.
    """
    train_blocks, test_blocks = [], []
    for _, (fmin, fmax) in bands.items():
        X_tr_b = bandpass_epochs(X_train, fmin=fmin, fmax=fmax, fs=fs)
        X_te_b = bandpass_epochs(X_test, fmin=fmin, fmax=fmax, fs=fs)

        cov_tr = compute_spd_covariances_scratch(X_tr_b, shrinkage=shrinkage)
        cov_te = compute_spd_covariances_scratch(X_te_b, shrinkage=shrinkage)

        ref_mean = riemannian_mean_scratch(cov_tr, max_iter=30, tol=1e-6)
        train_blocks.append(tangent_space_project_scratch(cov_tr, ref_mean))
        test_blocks.append(tangent_space_project_scratch(cov_te, ref_mean))
    return np.hstack(train_blocks), np.hstack(test_blocks)


def _eval(pipe, F_tr, y_tr, F_te, y_te, cv) -> dict[str, float]:
    cv_acc = float(np.mean(cross_val_score(pipe, F_tr, y_tr, cv=cv, scoring="accuracy")))
    pipe.fit(F_tr, y_tr)
    pred = pipe.predict(F_te)
    return {
        "cv_acc": cv_acc,
        "test_acc": float(accuracy_score(y_te, pred)),
        "test_kappa": float(cohen_kappa_score(y_te, pred)),
        "test_f1": float(f1_score(y_te, pred, average="macro")),
    }


def run_all_subjects(
    subjects: range = range(1, 10),
    shrinkage: float = 0.03,
    C: float = 0.05,
    bands: dict[str, tuple[float, float]] = SUB_BANDS_4,
) -> pd.DataFrame:
    records = []
    le = LabelEncoder()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for sid in subjects:
        print(f"Processing Subject {sid}/9...")
        X_tr, y_tr_s, X_te, y_te_s = load_subject_raw_epochs(subject_id=sid, fmin=4.0, fmax=38.0)

        # Slice to active motor imagery window (0.5s to 3.5s at 250 Hz)
        X_tr = X_tr[:, :, 125:875]
        X_te = X_te[:, :, 125:875]

        y_tr = le.fit_transform(y_tr_s)
        y_te = le.transform(y_te_s)

        clf = make_pipeline(StandardScaler(), SVC(C=C, kernel="linear", random_state=42))

        # 1. Euclidean SVM
        Fe_tr = extract_euclidean_cov_features(X_tr, shrinkage=shrinkage)
        Fe_te = extract_euclidean_cov_features(X_te, shrinkage=shrinkage)
        m_eucl = _eval(clf, Fe_tr, y_tr, Fe_te, y_te, cv)

        # 2. Single-Band (8-30 Hz) Riemannian SVM
        X_tr_8_30 = bandpass_epochs(X_tr, 8.0, 30.0)
        X_te_8_30 = bandpass_epochs(X_te, 8.0, 30.0)
        c_tr_8_30 = compute_spd_covariances_scratch(X_tr_8_30, shrinkage=shrinkage)
        c_te_8_30 = compute_spd_covariances_scratch(X_te_8_30, shrinkage=shrinkage)
        mean_8_30 = riemannian_mean_scratch(c_tr_8_30, max_iter=30, tol=1e-6)
        Fs_tr = tangent_space_project_scratch(c_tr_8_30, mean_8_30)
        Fs_te = tangent_space_project_scratch(c_te_8_30, mean_8_30)
        m_sb = _eval(clf, Fs_tr, y_tr, Fs_te, y_te, cv)

        # 3. Sub-Band Fusion (4-Band)
        Ff_tr, Ff_te = fit_transform_unaligned_subband_fusion(
            X_tr, X_te, bands=bands, shrinkage=shrinkage
        )
        m_fus = _eval(clf, Ff_tr, y_tr, Ff_te, y_te, cv)

        # 4. Sub-Band Fusion + Session Recentering
        Fa_tr, Fa_te = fit_transform_aligned_subband_fusion(
            X_tr, X_te, bands=bands, shrinkage=shrinkage
        )
        m_al = _eval(clf, Fa_tr, y_tr, Fa_te, y_te, cv)

        model_map = {
            "A1_Euclidean_SVM": m_eucl,
            "A2_SingleBand_Riemann_SVM": m_sb,
            "A3_SubBand_Fusion_4Band": m_fus,
            "A4_Aligned_SubBand_Fusion_4Band": m_al,
        }
        for model_name, metrics in model_map.items():
            records.append({"subject": f"S{sid:02d}", "model": model_name, **metrics})

    df = pd.DataFrame(records)
    df.to_csv("benchmark_9_subjects.csv", index=False)

    summary = df.groupby("model")[["cv_acc", "test_acc", "test_kappa", "test_f1"]].mean()
    print("\n=== Mean Performance Across All 9 Subjects (BCI Comp IV-2a) ===")
    print(
        summary.assign(
            cv_acc=lambda d: (d["cv_acc"] * 100).round(2).astype(str) + "%",
            test_acc=lambda d: (d["test_acc"] * 100).round(2).astype(str) + "%",
            test_kappa=lambda d: d["test_kappa"].round(4),
            test_f1=lambda d: (d["test_f1"] * 100).round(2).astype(str) + "%",
        ).to_string()
    )
    print("\nSaved full 9-subject results to 'benchmark_9_subjects.csv'.")
    return df


if __name__ == "__main__":
    run_all_subjects(shrinkage=0.03, C=0.05, bands=SUB_BANDS_4)