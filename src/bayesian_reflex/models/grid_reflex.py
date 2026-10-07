"""E7 exact-grid reference agent: the closed Bayesian reflex loop on a functional state-space belief.

Per trial: belief N(m, P) over f on a grid (exact functional Kalman filter, kernel-correlated drift Q = seta^2 * scale * K).
One step = choose the query (policy) -> predict -> standardized error z -> Bayes update with drift -> store z^2.
Error-driven plasticity (E7 protocol): scale_t = clip(mean of z^2 over the PREVIOUS <=5 steps, 1, 100); z_t itself is not used
for the same step's noise level (it depends on it). Batched over trials (covariance differs per trial once plasticity is on).
Policies: 'variance' = argmax of the predicted epistemic variance diag(P_pred); 'random' = indices supplied by the environment.
"""
import numpy as np


class GridReflexAgent:
    def __init__(self, n, K, sy_true, seta, policy="variance", plastic=True, sy_assumed=None, window=5, clip=(1.0, 100.0)):
        self.n, self.K, self.G = n, K, K.shape[0]
        self.sy_true, self.sy_a2 = sy_true, (sy_true if sy_assumed is None else sy_assumed) ** 2
        self.seta2, self.policy, self.plastic, self.window, self.clip = seta ** 2, policy, plastic, window, clip
        self.m, self.P = np.zeros((n, self.G)), np.tile(K, (n, 1, 1))
        self.z2_hist, self.ar = [], np.arange(n)

    def scale(self):
        if not self.plastic or not self.z2_hist:
            return np.ones(self.n)
        return np.clip(np.mean(self.z2_hist[-self.window:], axis=0), *self.clip)

    def step(self, f_t, eps_t, rand_idx_t):
        Pp = self.P + (self.seta2 * self.scale())[:, None, None] * self.K       # time update (drift, possibly inflated)
        i = np.argmax(np.diagonal(Pp, axis1=1, axis2=2), axis=1) if self.policy == "variance" else rand_idx_t
        y = f_t[self.ar, i] + self.sy_true * eps_t                                # environment answers the query
        mp = self.m[self.ar, i]
        s = Pp[self.ar, i, i] + self.sy_a2
        z = (y - mp) / np.sqrt(s)                                                  # standardized prediction error (before update)
        g = Pp[self.ar, :, i] / s[:, None]
        self.m = self.m + (y - mp)[:, None] * g
        self.P = Pp - g[:, :, None] * Pp[self.ar, i, :][:, None, :]
        self.z2_hist.append(z ** 2)
        return i, z

    def posterior_sd(self):
        return np.sqrt(np.maximum(np.diagonal(self.P, axis1=1, axis2=2), 1e-12))
