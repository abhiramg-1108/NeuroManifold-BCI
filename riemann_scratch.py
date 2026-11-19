import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann


def _sym_matrix_func(mat: np.ndarray, func, eps: float = 1e-12) -> np.ndarray:
    """Applies a scalar function `func` to the eigenvalues of a symmetric matrix."""
    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals_safe = np.clip(eigvals, eps, None)
    mapped_vals = func(eigvals_safe)
    out = (eigvecs * mapped_vals) @ eigvecs.T
    return 0.5 * (out + out.T)


def matrix_power_spd(mat: np.ndarray, power: float, eps: float = 1e-12) -> np.ndarray:
    """Computes mat ** power for an SPD matrix via eigendecomposition."""
    return _sym_matrix_func(mat, lambda lam: lam ** power, eps=eps)


def matrix_log_spd(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Computes matrix logarithm log(mat) for an SPD matrix."""
    return _sym_matrix_func(mat, np.log, eps=eps)


def matrix_exp_sym(mat: np.ndarray) -> np.ndarray:
    """Computes matrix exponential exp(mat) for a real symmetric matrix."""
    sym = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(sym)
    out = (eigvecs * np.exp(eigvals)) @ eigvecs.T
    return 0.5 * (out + out.T)


def compute_spd_covariances_scratch(X: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    """
    Computes regularized SPD covariance matrices from epochs X of shape (n_trials, n_channels, n_times).
    Uses Ledoit-Wolf style shrinkage toward scaled identity to guarantee positive definiteness.
    """
    n_trials, n_channels, n_times = X.shape
    covs = np.empty((n_trials, n_channels, n_channels), dtype=np.float64)

    eye = np.eye(n_channels, dtype=np.float64)
    for i in range(n_trials):
        trial = X[i] - np.mean(X[i], axis=1, keepdims=True)
        emp_cov = (trial @ trial.T) / max(n_times - 1, 1)
        scale = np.trace(emp_cov) / n_channels
        reg_cov = (1.0 - shrinkage) * emp_cov + shrinkage * scale * eye
        covs[i] = 0.5 * (reg_cov + reg_cov.T)

    return covs


def riemannian_mean_scratch(
    covs: np.ndarray, max_iter: int = 50, tol: float = 1e-7
) -> np.ndarray:
    """
    Computes the Affine-Invariant Riemannian (Fréchet) mean of SPD matrices
    using fixed-point / geodesic gradient descent on the SPD manifold.
    """
    # Initialize with Euclidean mean (guaranteed SPD)
    P = np.mean(covs, axis=0)
    P = 0.5 * (P + P.T)

    for _ in range(max_iter):
        P_half = matrix_power_spd(P, 0.5)
        P_inv_half = matrix_power_spd(P, -0.5)

        # Average log-mapped matrices in tangent space at P
        tangent_sum = np.zeros_like(P)
        for C in covs:
            whitened = P_inv_half @ C @ P_inv_half
            tangent_sum += matrix_log_spd(whitened)

        delta_tangent = tangent_sum / len(covs)
        # Update along geodesic: P <- P^{1/2} exp(delta_tangent) P^{1/2}
        P_new = P_half @ matrix_exp_sym(delta_tangent) @ P_half
        P_new = 0.5 * (P_new + P_new.T)

        step_norm = np.linalg.norm(delta_tangent, ord="fro")
        P = P_new
        if step_norm < tol:
            break

    return P


def tangent_space_project_scratch(covs: np.ndarray, ref_mean: np.ndarray) -> np.ndarray:
    """
    Projects SPD matrices onto the tangent space at `ref_mean` and returns
    upper-triangular half-vectorized features of shape (n_trials, C*(C+1)//2).
    Off-diagonal entries are scaled by sqrt(2) to preserve the Riemannian inner product.
    """
    n_trials, n_channels, _ = covs.shape
    P_half = matrix_power_spd(ref_mean, 0.5)
    P_inv_half = matrix_power_spd(ref_mean, -0.5)

    upper_idx = np.triu_indices(n_channels)
    # Mask for strict off-diagonal elements
    off_diag_mask = upper_idx[0] != upper_idx[1]
    scale_weights = np.ones(len(upper_idx[0]), dtype=np.float64)
    scale_weights[off_diag_mask] = np.sqrt(2.0)

    feat_dim = len(upper_idx[0])
    features = np.empty((n_trials, feat_dim), dtype=np.float64)

    for i, C in enumerate(covs):
        whitened = P_inv_half @ C @ P_inv_half
        log_w = matrix_log_spd(whitened)
        # Tangent vector at ref_mean: T_P M
        tangent_mat = P_half @ log_w @ P_half
        features[i] = tangent_mat[upper_idx] * scale_weights

    return features


if __name__ == "__main__":
    from data_loader import load_subject_raw_epochs

    X_train, _, X_test, _ = load_subject_raw_epochs(subject_id=1, fmin=8.0, fmax=30.0)

    # 1. Covariance comparison against pyRiemann (OAS estimator)
    cov_scratch = compute_spd_covariances_scratch(X_train, shrinkage=0.1)
    cov_ref = Covariances(estimator="oas").transform(X_train)
    print(f"Covariance feature shape: {cov_scratch.shape}")

    # 2. Riemannian mean verification against pyRiemann
    mean_scratch = riemannian_mean_scratch(cov_scratch)
    mean_ref = mean_riemann(cov_scratch)
    mean_diff = np.linalg.norm(mean_scratch - mean_ref, ord="fro")
    print(f"Riemannian mean Frobenius diff vs pyRiemann: {mean_diff:.2e}")

    # 3. Tangent space projection check
    ts_scratch = tangent_space_project_scratch(cov_scratch, mean_scratch)
    ts_ref = TangentSpace(metric="riemann").fit_transform(cov_scratch)
    print(f"Scratch tangent space shape: {ts_scratch.shape} | pyRiemann shape: {ts_ref.shape}")