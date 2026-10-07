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


class InducingGridAgent:
    """E7 confirmation agent on the PC-FSVI STAND-IN: whitened inducing posterior q(u)=N(m, S) per trial (S batched, since
    plasticity makes it trial-specific). Same loop, policy, plasticity rule and noise handling as GridReflexAgent; drift acts as
    S += seta^2 * scale * I (whitened). Predictive variance includes the Nystrom residual; the update ignores it (as the stand-in does)."""
    def __init__(self, n, sy_true, seta, policy="variance", plastic=True, sy_assumed=None, M=20, window=5, clip=(1.0, 100.0), grid=None):
        from .function_space import PCFSVI
        self.n, self.sy_true, self.sy_a2 = n, sy_true, (sy_true if sy_assumed is None else sy_assumed) ** 2
        self.seta2, self.policy, self.plastic, self.window, self.clip, self.M = seta ** 2, policy, plastic, window, clip, M
        self.Phi = PCFSVI(M=M, sy=sy_true).phi(grid)                      # (M, G)
        self.resid = np.maximum(1.0 - (self.Phi ** 2).sum(0), 0.0)
        self.m, self.S = np.zeros((n, M)), np.tile(np.eye(M), (n, 1, 1))
        self.z2_hist, self.ar = [], np.arange(n)

    scale = GridReflexAgent.scale

    def _var(self, S):
        return self.resid[None, :] + np.einsum("mg,nmk,kg->ng", self.Phi, S, self.Phi)

    def step(self, f_t, eps_t, rand_idx_t):
        Sp = self.S + (self.seta2 * self.scale())[:, None, None] * np.eye(self.M)
        vg = self._var(Sp)
        i = np.argmax(vg, axis=1) if self.policy == "variance" else rand_idx_t
        y = f_t[self.ar, i] + self.sy_true * eps_t
        phi = self.Phi[:, i].T
        mp = np.einsum("nm,nm->n", self.m, phi)
        z = (y - mp) / np.sqrt(vg[self.ar, i] + self.sy_a2)
        Sphi = np.einsum("nmk,nk->nm", Sp, phi)
        den = (phi * Sphi).sum(1) + self.sy_a2
        self.m = self.m + ((y - mp) / den)[:, None] * Sphi
        self.S = Sp - Sphi[:, :, None] * Sphi[:, None, :] / den[:, None, None]
        self.z2_hist.append(z ** 2)
        return i, z

    def mean_sd(self):
        return self.m @ self.Phi, np.sqrt(np.maximum(self._var(self.S), 1e-12))


class RealGridAgent:
    """E7 confirmation agent on the user's REAL FSVI + pc_infer (one RealPCFSVI per trial, n_iters=5, lr=1: equal to 150 in E4-real).
    Drift/plasticity: each update runs with sigma_eta = seta * sqrt(scale), i.e. prior_cov = S + seta^2 * scale * K_zz.
    Prediction and action use the same time-updated covariance, so the loop matches GridReflexAgent."""
    def __init__(self, n, sy_true, seta, policy="variance", plastic=True, sy_assumed=None, M=20, window=5, clip=(1.0, 100.0), grid=None, n_iters=5):
        from .pcfsvi_adapter import RealPCFSVI
        sy_a = sy_true if sy_assumed is None else sy_assumed
        self.reals = [RealPCFSVI(M=M, sy=sy_a, n_iters=n_iters, lr=1.0) for _ in range(n)]
        mdl = self.reals[0].model
        Kxz = mdl._Kxz(np.asarray(grid, float).reshape(-1, 1))
        self.A, self.Kzz = Kxz @ mdl.Kzz_inv, mdl.Kzz
        self.resid = mdl.kernel_variance - (self.A * Kxz).sum(1)
        self.n, self.sy_true, self.sy_a2, self.seta, self.seta2 = n, sy_true, sy_a ** 2, seta, seta ** 2
        self.policy, self.plastic, self.window, self.clip = policy, plastic, window, clip
        self._grid_x = np.asarray(grid, float)
        self.z2_hist, self.ar = [], np.arange(n)

    scale = GridReflexAgent.scale

    def _stack(self):
        return np.stack([r.m for r in self.reals]), np.stack([r.S for r in self.reals])

    def _var(self, S):
        return self.resid[None, :] + np.einsum("gm,nmk,gk->ng", self.A, S, self.A)

    def step(self, f_t, eps_t, rand_idx_t):
        scale = self.scale()
        m, S = self._stack()
        vg = np.maximum(self._var(S + (self.seta2 * scale)[:, None, None] * self.Kzz), 1e-12)
        i = np.argmax(vg, axis=1) if self.policy == "variance" else rand_idx_t
        y = f_t[self.ar, i] + self.sy_true * eps_t
        mp = (m @ self.A.T)[self.ar, i]
        z = (y - mp) / np.sqrt(vg[self.ar, i] + self.sy_a2)
        gx = np.asarray(self._grid_x)[i]
        for j, r in enumerate(self.reals):
            r.sigma_eta = self.seta * np.sqrt(scale[j])
            r.update(y[j], gx[j])
        self.z2_hist.append(z ** 2)
        return i, z

    def mean_sd(self):
        m, S = self._stack()
        return m @ self.A.T, np.sqrt(np.maximum(self._var(S), 1e-12))
