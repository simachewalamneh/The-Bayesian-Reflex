"""E4 function-space beliefs over f(x) with an RBF prior (hyperparameters fixed and shared).

ExactGP : exact GP oracle. Sequential rank-one Cholesky append, O(t^2) per update (batch refit is O(t^3)).
PCFSVI  : MINIMAL STAND-IN for PC-FSVI, not the pc-fsvi-continual-learning code. Function-space variational
          posterior q(u)=N(m,S) over M inducing values, in whitened coordinates (prior N(0,I)); each streaming
          update is a predictive-coding settling: w <- w + eta*(S phi * eps_y - (w - m_prev)), with precision-weighted
          error eps_y = (y - phi^T w)/sigma_y^2. For a Gaussian likelihood the fixed point is the conjugate (Kalman)
          update, so the stand-in equals the optimal sparse variational posterior. Cost O(M^2) per update, independent of t.
Both expose predict_f(xs) -> (mean, var_f) (latent function) and predict(xs) -> (mean, var_f + sigma_y^2).
sample() draws MARGINAL (independent per-point) samples, not joint function draws.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular

from ..predictive_coding import PredictiveCodingLayer
from .base import GenerativeModel


def rbf(a, b, sf=1.0, ell=1.0):
    return sf ** 2 * np.exp(-0.5 * (np.asarray(a)[:, None] - np.asarray(b)[None, :]) ** 2 / ell ** 2)


class _Marginal:
    def predict(self, query=None):
        mu, v = self.predict_f(query)
        return mu, v + self.sy ** 2

    def sample(self, n, rng, query=None):
        if query is None:
            raise ValueError("query inputs required")
        mu, v = self.predict_f(query)
        return mu[:, None] + np.sqrt(v)[:, None] * rng.standard_normal((len(mu), n))


class ExactGP(_Marginal, GenerativeModel):
    def __init__(self, sf=1.0, ell=1.0, sy=0.2, capacity=1000):
        self.sf, self.ell, self.sy = sf, ell, sy
        self.X, self.y = np.empty(capacity), np.empty(capacity)
        self.L, self.t = np.zeros((capacity, capacity)), 0

    def update(self, observation, context=None):
        t, x = self.t, float(context)
        if t == 0:
            self.L[0, 0] = np.sqrt(self.sf ** 2 + self.sy ** 2)
        else:
            kx = rbf(self.X[:t], [x], self.sf, self.ell)[:, 0]
            l21 = solve_triangular(self.L[:t, :t], kx, lower=True, check_finite=False)
            self.L[t, :t] = l21
            self.L[t, t] = np.sqrt(self.sf ** 2 + self.sy ** 2 - l21 @ l21)
        self.X[t], self.y[t] = x, observation
        self.t += 1

    def predict_f(self, xs):
        t, L = self.t, self.L[:self.t, :self.t]
        Ks = rbf(self.X[:t], xs, self.sf, self.ell)
        v = solve_triangular(L, Ks, lower=True, check_finite=False)
        a = solve_triangular(L, self.y[:t], lower=True, check_finite=False)
        alpha = solve_triangular(L.T, a, lower=False, check_finite=False)
        return Ks.T @ alpha, self.sf ** 2 - (v ** 2).sum(0)


def batch_gp(X, y, xs, sf=1.0, ell=1.0, sy=0.2):
    """Batch exact GP (O(t^3)); returns (mean, var_f) on xs."""
    c = cho_factor(rbf(X, X, sf, ell) + sy ** 2 * np.eye(len(X)), lower=True)
    Ks = rbf(X, xs, sf, ell)
    return Ks.T @ cho_solve(c, y), sf ** 2 - (Ks * cho_solve(c, Ks)).sum(0)


class PCFSVI(_Marginal, GenerativeModel):
    def __init__(self, M=20, domain=(-3.0, 3.0), sf=1.0, ell=1.0, sy=0.2, n_iters=30, jitter=1e-6,
                 sigma_eta=0.0, forgetting=None):
        self.sf, self.ell, self.sy, self.n_iters, self.M = sf, ell, sy, n_iters, M
        self.sigma_eta, self.forgetting = sigma_eta, forgetting   # E5-B drift: S += sigma_eta^2 I, and/or S /= lambda
        self.Z = np.linspace(*domain, M)
        self.Lz = cholesky(rbf(self.Z, self.Z, sf, ell) + jitter * np.eye(M), lower=True)
        self.m, self.S = np.zeros(M), np.eye(M)
        self.pc = PredictiveCodingLayer()

    def phi(self, x):                                   # (M, n) whitened features
        return solve_triangular(self.Lz, rbf(self.Z, np.atleast_1d(x), self.sf, self.ell), lower=True)

    def update(self, observation, context=None):
        if self.sigma_eta:                              # time update (functional drift, whitened coordinates)
            self.S = self.S + self.sigma_eta ** 2 * np.eye(self.M)
        if self.forgetting:
            self.S = self.S / self.forgetting
        phi = self.phi(float(context))[:, 0]
        s2, Sphi = self.sy ** 2, self.S @ phi
        c = phi @ Sphi / s2
        eta = 0.5 / (1.0 + c)                           # settling contracts by 0.5 per iteration
        delta = np.zeros(self.M)
        for _ in range(self.n_iters):                   # predictive-coding settling
            eps = self.pc.precision_weighted_error(observation, phi @ (self.m + delta), s2)
            delta += eta * (Sphi * eps - delta)
        self.m = self.m + delta
        self.S = self.S - np.outer(Sphi, Sphi) / (s2 + phi @ Sphi)
        self.S = 0.5 * (self.S + self.S.T)

    def batch_posterior(self, X, y):
        """Closed-form conjugate posterior of q(u) given all data (reference fixed point)."""
        Phi = self.phi(np.asarray(X))
        Lam = np.eye(self.M) + Phi @ Phi.T / self.sy ** 2
        return np.linalg.solve(Lam, Phi @ y / self.sy ** 2), np.linalg.inv(Lam)

    def predict_f(self, xs):
        phi = self.phi(np.asarray(xs))
        var = self.sf ** 2 - (phi ** 2).sum(0) + (phi * (self.S @ phi)).sum(0)
        return phi.T @ self.m, np.maximum(var, 1e-12)
