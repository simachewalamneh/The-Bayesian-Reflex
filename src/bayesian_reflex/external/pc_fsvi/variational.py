import numpy as np
from scipy.optimize import minimize
from .utils import rbf_kernel


def _pack(m, L):
    M = m.shape[0]
    tril_idx = np.tril_indices(M, k=-1)
    diag = np.diag(L)
    return np.concatenate([m, np.log(np.maximum(diag, 1e-8)), L[tril_idx]])


def _unpack(theta, M):
    m = theta[:M]
    log_diag = theta[M : 2 * M]
    off_diag = theta[2 * M :]
    L = np.zeros((M, M))
    tril_idx = np.tril_indices(M, k=-1)
    L[tril_idx] = off_diag
    np.fill_diagonal(L, np.exp(log_diag))
    return m, L


class FSVI:

    def __init__(self, Z, lengthscale=1.0, kernel_variance=1.0, noise_std=0.1):
        self.Z = np.atleast_2d(Z)
        self.M = len(self.Z)
        self.lengthscale = lengthscale
        self.kernel_variance = kernel_variance
        self.noise_std = noise_std
        self.Kzz = rbf_kernel(self.Z, self.Z, lengthscale, kernel_variance) + 1e-6 * np.eye(self.M)
        self.Kzz_inv = np.linalg.inv(self.Kzz)
        # variational parameters, initialised to the GP prior
        self.m = np.zeros(self.M)
        self.L = np.linalg.cholesky(self.Kzz)

    # ---- GP-conditional machinery -------------------------------------
    def _Kxz(self, X):
        return rbf_kernel(X, self.Z, self.lengthscale, self.kernel_variance)

    def predict(self, X, m=None, L=None):
        """q(f(x)) mean & variance via the exact GP conditional on u."""
        if m is None:
            m = self.m
        if L is None:
            L = self.L
        S = L @ L.T
        Kxz = self._Kxz(X)
        A = Kxz @ self.Kzz_inv  # (N, M)
        mean = A @ m
        Kxx_diag = self.kernel_variance * np.ones(len(X))
        var = Kxx_diag - np.sum(A * Kxz, axis=1) + np.sum((A @ S) * A, axis=1)
        var = np.maximum(var, 1e-10)
        return mean, var

    # ---- Free energy ----------------------------------------------------
    def free_energy(self, theta, X, y, beta=1.0, prior_mean=None, prior_cov=None):
        m, L = _unpack(theta, self.M)
        S = L @ L.T
        if prior_mean is None:
            prior_mean = np.zeros(self.M)
        if prior_cov is None:
            prior_cov = self.Kzz

        mean_f, var_f = self.predict(X, m, L)
        sigma_n2 = self.noise_std ** 2
        # expected negative log-likelihood (closed form for Gaussian lik.)
        nll = 0.5 * np.sum(
            np.log(2 * np.pi * sigma_n2) + ((y - mean_f) ** 2 + var_f) / sigma_n2
        )

        prior_cov_inv = np.linalg.inv(prior_cov + 1e-8 * np.eye(self.M))
        sign0, logdet0 = np.linalg.slogdet(prior_cov)
        sign1, logdet1 = np.linalg.slogdet(S + 1e-10 * np.eye(self.M))
        diff = prior_mean - m
        kl = 0.5 * (
            np.trace(prior_cov_inv @ S)
            + diff @ prior_cov_inv @ diff
            - self.M
            + logdet0
            - logdet1
        )
        return nll + beta * kl

    def free_energy_and_grad(self, theta, X, y, beta=1.0, prior_mean=None, prior_cov=None):

        m, L = _unpack(theta, self.M)
        S = L @ L.T
        if prior_mean is None:
            prior_mean = np.zeros(self.M)
        if prior_cov is None:
            prior_cov = self.Kzz
        sigma_n2 = self.noise_std ** 2

        Kxz = self._Kxz(X)
        A = Kxz @ self.Kzz_inv  # (N, M)
        mean_f = A @ m
        Kxx_diag = self.kernel_variance * np.ones(len(X))
        var_f = np.maximum(Kxx_diag - np.sum(A * Kxz, axis=1) + np.sum((A @ S) * A, axis=1), 1e-10)

        nll = 0.5 * np.sum(np.log(2 * np.pi * sigma_n2) + ((y - mean_f) ** 2 + var_f) / sigma_n2)

        prior_cov_inv = np.linalg.inv(prior_cov + 1e-8 * np.eye(self.M))
        _, logdet0 = np.linalg.slogdet(prior_cov)
        _, logdet1 = np.linalg.slogdet(S + 1e-10 * np.eye(self.M))
        diff = prior_mean - m
        kl = 0.5 * (
            np.trace(prior_cov_inv @ S) + diff @ prior_cov_inv @ diff - self.M + logdet0 - logdet1
        )
        fval = nll + beta * kl

        # --- gradients ---
        eps = y - mean_f
        grad_m = -(A.T @ (eps / sigma_n2)) + beta * (prior_cov_inv @ (m - prior_mean))
        grad_S = 0.5 / sigma_n2 * (A.T @ A) + 0.5 * beta * (prior_cov_inv - np.linalg.inv(S + 1e-10 * np.eye(self.M)))
        grad_L = 2.0 * grad_S @ L  # dF/dL from dF/dS via S = L L^T (G symmetric)

        M = self.M
        tril_idx = np.tril_indices(M, k=-1)
        grad_theta = np.zeros_like(theta)
        grad_theta[:M] = grad_m
        grad_theta[M:2 * M] = np.diag(grad_L) * np.diag(L)  # chain rule: L_ii = exp(log_diag_i)
        grad_theta[2 * M:] = grad_L[tril_idx]
        return fval, grad_theta

    def fit(self, X, y, beta=1.0, prior_mean=None, prior_cov=None, maxiter=3000):
        theta0 = _pack(self.m, self.L)
        res = minimize(
            self.free_energy_and_grad,
            theta0,
            args=(X, y, beta, prior_mean, prior_cov),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": maxiter},
        )
        self.m, self.L = _unpack(res.x, self.M)
        return res

    def posterior(self):
        """Return current (mean, covariance) of q(u), for chaining across tasks."""
        return self.m.copy(), (self.L @ self.L.T).copy()
