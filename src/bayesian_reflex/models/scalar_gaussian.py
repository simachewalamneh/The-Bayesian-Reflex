"""E1 model: conjugate Gaussian belief over a scalar theta.

Prior theta ~ N(mu0, s0^2); likelihood y_t = theta + eps, eps ~ N(0, sy^2).
Array-valued so many independent trials run in parallel (one belief each).
"""
import numpy as np

from .base import GenerativeModel


class ScalarGaussianBelief(GenerativeModel):
    def __init__(self, mu0=0.0, s0=1.0, sy=0.5, n_trials=1):
        self.sy2 = sy ** 2
        self.mu = np.full(n_trials, float(mu0))
        self.prec = np.full(n_trials, 1.0 / s0 ** 2)

    @property
    def var(self):
        return 1.0 / self.prec

    @property
    def sd(self):
        return np.sqrt(self.var)

    def update(self, observation, context=None):
        new_prec = self.prec + 1.0 / self.sy2
        self.mu = (self.prec * self.mu + observation / self.sy2) / new_prec
        self.prec = new_prec

    def predict(self, query=None):
        return self.mu.copy(), self.var + self.sy2

    def sample(self, n, rng):
        return self.mu[:, None] + self.sd[:, None] * rng.standard_normal((len(self.mu), n))
