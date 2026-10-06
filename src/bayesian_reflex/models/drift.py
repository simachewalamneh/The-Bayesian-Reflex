"""E5-A model: scalar state-space belief with explicit process noise (Kalman filter in belief form).

theta_t = theta_{t-1} + eta_t (eta ~ N(0, sigma_eta^2)),  y_t = theta_t + eps_t (eps ~ N(0, sy^2)).
Time update: var_pred = var + sigma_eta^2.  Measurement update: standard Gaussian Bayes.
forgetting (VCL-style tempering q_t ∝ p(y|theta) q_{t-1}^lambda) is the same filter with var_pred = var / lambda:
  forgetting=None      -> process noise sigma_eta (Kalman)
  forgetting='matched' -> lambda_t = var/(var+sigma_eta^2) each step (identical to Kalman, algebraically)
  forgetting=float     -> constant lambda
sigma_eta = 0 reproduces the static belief of E1.
"""
import numpy as np

from .base import GenerativeModel


class ScalarDriftBelief(GenerativeModel):
    def __init__(self, mu0=0.0, s0=1.0, sy=0.5, sigma_eta=0.0, n_trials=1, forgetting=None):
        self.sy2, self.seta2, self.forgetting = sy ** 2, sigma_eta ** 2, forgetting
        self.mu = np.full(n_trials, float(mu0))
        self.var = np.full(n_trials, s0 ** 2)

    @property
    def sd(self):
        return np.sqrt(self.var)

    def _var_pred(self):
        if self.forgetting is None:
            return self.var + self.seta2
        lam = self.var / (self.var + self.seta2) if self.forgetting == "matched" else float(self.forgetting)
        return self.var / lam

    def predict(self, query=None):
        """One-step-ahead predictive for the next y, INCLUDING drift (does not mutate state)."""
        return self.mu.copy(), self._var_pred() + self.sy2

    def update(self, observation, context=None):
        vp = self._var_pred()
        gain = vp / (vp + self.sy2)
        self.mu = self.mu + gain * (observation - self.mu)
        self.var = vp * self.sy2 / (vp + self.sy2)

    def sample(self, n, rng):
        return self.mu[:, None] + self.sd[:, None] * rng.standard_normal((len(self.mu), n))
