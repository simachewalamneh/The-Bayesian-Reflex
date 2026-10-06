import numpy as np
from .utils import rbf_kernel


class ExactGP:
    def __init__(self, lengthscale=1.0, variance=1.0, noise_std=0.1):
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise_std = noise_std
        self.X = None
        self.y = None
        self._L = None
        self._alpha = None

    def kernel(self, X1, X2):
        return rbf_kernel(X1, X2, self.lengthscale, self.variance)

    def fit(self, X, y):
        """Store data and precompute Cholesky factor of K + sigma_n^2 I."""
        self.X = np.atleast_2d(X)
        self.y = np.asarray(y).reshape(-1)
        K = self.kernel(self.X, self.X) + (self.noise_std ** 2) * np.eye(len(self.X))
        self._L = np.linalg.cholesky(K + 1e-8 * np.eye(len(self.X)))
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, self.y))
        return self

    def predict(self, Xs):
        """Return posterior mean and variance at test points Xs.
        mean = K(Xs,X) alpha
        var  = k(Xs,Xs) - v^T v,  v = L^{-1} K(X,Xs)
        """
        Xs = np.atleast_2d(Xs)
        if self.X is None:
            # prior predictive
            mean = np.zeros(len(Xs))
            var = np.full(len(Xs), self.variance)
            return mean, var
        Ks = self.kernel(self.X, Xs)  # (N, M)
        mean = Ks.T @ self._alpha
        v = np.linalg.solve(self._L, Ks)
        var = self.variance - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)
        return mean, var

    def sample_prior(self, Xs, n_samples=5, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        K = self.kernel(Xs, Xs) + 1e-8 * np.eye(len(Xs))
        L = np.linalg.cholesky(K)
        z = rng.normal(size=(len(Xs), n_samples))
        return (L @ z).T  # (n_samples, M)

    def neg_log_marginal_likelihood(self):
        """-log p(y|X): standard GP evidence, used only for diagnostics."""
        n = len(self.y)
        term1 = 0.5 * self.y @ self._alpha
        term2 = np.sum(np.log(np.diag(self._L)))
        term3 = 0.5 * n * np.log(2 * np.pi)
        return term1 + term2 + term3
