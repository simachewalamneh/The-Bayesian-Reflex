"""Adapter exposing the user's REAL FSVI + pc_infer (src/bayesian_reflex/external/pc_fsvi) through the
update / predict_f interface used by E4. Chaining follows pc_fsvi_continual: a chunk of data is one 'task',
the previous posterior is the prior, pc_infer runs, then (m, chol(S)) are stored. chunk size 1 = streaming."""
import numpy as np

from ..external.pc_fsvi import FSVI, pc_infer
from .base import GenerativeModel


def _safe_chol(S, jitter=1e-8):          # same logic as continual_learning._safe_chol
    S = 0.5 * (S + S.T)
    for k in range(12):
        try:
            return np.linalg.cholesky(S + jitter * (10.0 ** k) * np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("covariance not PD even with jitter")


class RealPCFSVI(GenerativeModel):
    def __init__(self, M=20, domain=(-3.0, 3.0), sf=1.0, ell=1.0, sy=0.2, n_iters=150, lr=1.0, beta=1.0, sigma_eta=0.0):
        self.sy, self.n_iters, self.lr, self.beta, self.sigma_eta = sy, n_iters, lr, beta, sigma_eta
        self.model = FSVI(np.linspace(*domain, M).reshape(-1, 1), lengthscale=ell, kernel_variance=sf, noise_std=sy)
        self.prior_mean, self.prior_cov = np.zeros(M), self.model.Kzz.copy()

    @property
    def m(self):
        return self.model.m

    @property
    def S(self):
        return self.model.L @ self.model.L.T

    def update_batch(self, X, y):
        X = np.asarray(X, float).reshape(-1, 1)
        pcov = self.prior_cov + self.sigma_eta ** 2 * self.model.Kzz if self.sigma_eta else self.prior_cov   # E5-B drift time update
        m, S, _ = pc_infer(self.model, X, np.asarray(y, float), self.prior_mean, pcov,
                           beta=self.beta, n_iters=self.n_iters, lr=self.lr)
        self.model.m, self.model.L = m, _safe_chol(S)
        self.prior_mean, self.prior_cov = self.model.posterior()

    def update(self, observation, context=None):
        self.update_batch([float(context)], [float(observation)])

    def predict_f(self, xs):
        return self.model.predict(np.asarray(xs, float).reshape(-1, 1))

    def predict(self, query=None):
        mu, v = self.predict_f(query)
        return mu, v + self.sy ** 2

    def sample(self, n, rng, query=None):
        mu, v = self.predict_f(query)
        return mu[:, None] + np.sqrt(v)[:, None] * rng.standard_normal((len(mu), n))
