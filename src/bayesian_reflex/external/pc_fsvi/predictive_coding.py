import numpy as np


def _A_matrix(fsvi, X):
    Kxz = fsvi._Kxz(X)
    return Kxz @ fsvi.Kzz_inv


def pc_infer(
    fsvi,
    X,
    y,
    prior_mean,
    prior_cov,
    beta=1.0,
    n_iters=50,
    lr=0.1,
    adaptive_precision=False,
    precision_override=None,
):
    M = fsvi.M
    A = _A_matrix(fsvi, X)  # (N, M), fixed given current inducing points
    prior_cov_inv = np.linalg.inv(prior_cov + 1e-8 * np.eye(M))
    sigma_n2 = fsvi.noise_std ** 2

    m = fsvi.m.copy()
    S = (fsvi.L @ fsvi.L.T).copy()
    history = {"error_norm": [], "free_energy_proxy": []}

    for _ in range(n_iters):
        # --- prediction & prediction error (Sec. 5) ---
        mean_f = A @ m
        eps = y - mean_f

        # --- precision weighting (Sec. 8-9) ---
        if precision_override is not None:
            Pi = np.broadcast_to(np.asarray(precision_override, dtype=float), (len(X),)).copy()
        elif adaptive_precision:
            var_f = np.maximum(
                fsvi.kernel_variance - np.sum(A * fsvi._Kxz(X), axis=1)
                + np.sum((A @ S) * A, axis=1),
                1e-10,
            )
            Pi = 1.0 / (sigma_n2 + var_f)
        else:
            Pi = np.full(len(X), 1.0 / sigma_n2)

        # --- closed-form conjugate precision/covariance update (exact) ---
        Lambda = beta * prior_cov_inv + (A.T * Pi) @ A
        S = np.linalg.inv(Lambda + 1e-8 * np.eye(M))

        grad_m = -(A.T @ (Pi * eps)) + beta * (prior_cov_inv @ (m - prior_mean))
        m = m - lr * (S @ grad_m)

        history["error_norm"].append(float(np.sqrt(np.mean(eps ** 2))))

    return m, S, history


def closed_form_optimum(fsvi, X, y, prior_mean, prior_cov, beta=1.0):

    M = fsvi.M
    A = _A_matrix(fsvi, X)
    prior_cov_inv = np.linalg.inv(prior_cov + 1e-8 * np.eye(M))
    Pi = 1.0 / (fsvi.noise_std ** 2)
    Lambda = beta * prior_cov_inv + Pi * (A.T @ A)
    S = np.linalg.inv(Lambda + 1e-8 * np.eye(M))
    b = Pi * (A.T @ y) + beta * (prior_cov_inv @ prior_mean)
    m = S @ b
    return m, S
