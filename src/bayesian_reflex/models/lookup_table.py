"""E4b: the look-up table (LUT) emulator of paper 2605.02825 sec 5 / Ghosh et al. 1108.3262 sec 4, plus exact references.

GP emulator eta ~ GP(0, k) (zero mean, known scale), noise-free training D = eta(X_d). Dynamic sequence x_t = eta(v_t) (+ nugget),
v_t = input_fn(t, x_{t-1}).  LUT: fixed grid G*, auxiliary D* = eta(G*), one-time precomputation of the (D, D*) covariance
(Cholesky factor, and an explicit inverse for the T1 integrity test); Step 1 x_1 ~ [eta(v_1)|D]; Step 2 D* ~ [D*|D, eta(v_1)];
Step 3 x_t ~ N(m(v_t), r(v_t) + nugget) given (D, D*) only (Markov approximation).  Exact references condition on all past states.
"""
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular


def _pairs(A, B, ell):
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    return (((A[:, None, :] - B[None, :, :]) / np.broadcast_to(np.asarray(ell, float), (A.shape[1],))) ** 2).sum(-1)


def gauss_kernel(A, B, ell=1.0, sf=1.0):
    return sf ** 2 * np.exp(-0.5 * _pairs(A, B, ell))


def exp_kernel(A, B, ell=1.0, sf=1.0):
    return sf ** 2 * np.exp(-np.sqrt(_pairs(A, B, ell)))


def kl_gauss(m_p, C_p, m_q, C_q, jit=1e-9):
    """KL(N(m_p,C_p) || N(m_q,C_q)) with the same jitter added to both covariances."""
    k = len(m_p)
    Cp, Cq = C_p + jit * np.eye(k), C_q + jit * np.eye(k)
    L = np.linalg.cholesky(Cq)
    d = solve_triangular(L, m_p - m_q, lower=True)
    M = solve_triangular(L, Cp, lower=True)
    tr = np.trace(solve_triangular(L.T, M, lower=False))
    return 0.5 * (tr + d @ d - k + 2 * np.log(np.diag(L)).sum() - np.linalg.slogdet(Cp)[1])


class LookupTableEmulator:
    def __init__(self, Xd, yd, grid, kernel, jitter=1e-8, nugget=0.0, sf=1.0, solver="chol"):
        self.Xd, self.yd, self.G, self.k, self.jit, self.nug, self.sf2, self.solver = (
            np.atleast_2d(Xd), np.asarray(yd, float), np.atleast_2d(grid), kernel, jitter, nugget, sf ** 2, solver)
        self.n, self.N = len(self.Xd), len(self.G)
        self.P = np.vstack([self.Xd, self.G])
        self.A = kernel(self.P, self.P) + jitter * np.eye(self.n + self.N)        # covariance of (D, D*), formed once
        self.cho = cho_factor(self.A, lower=True)                                 # one-time precomputation
        self.L = self.cho[0]
        self.Ainv = cho_solve(self.cho, np.eye(self.n + self.N)); self.Ainv = 0.5 * (self.Ainv + self.Ainv.T)
        self.Kdd = kernel(self.Xd, self.Xd) + jitter * np.eye(self.n)
        self.cdd = cho_factor(self.Kdd, lower=True)

    def integrity(self):
        """T1: relative residual of the explicit inverse, and of the Cholesky solve."""
        I = np.eye(self.n + self.N)
        return (np.linalg.norm(self.A @ self.Ainv - I) / np.sqrt(len(I)),
                np.linalg.norm(self.A @ cho_solve(self.cho, I) - I) / np.sqrt(len(I)), np.linalg.cond(self.A))

    def posterior_given_D(self, Q):
        """Mean and covariance of eta at Q given D (exact GP)."""
        Kqd = self.k(Q, self.Xd)
        return Kqd @ cho_solve(self.cdd, self.yd), self.k(Q, Q) - Kqd @ cho_solve(self.cdd, Kqd.T)

    def prepare(self, v1):
        v1 = np.atleast_2d(v1)
        if np.min(np.sqrt(_pairs(v1, self.G, 1.0))) < 1e-6:
            raise ValueError("v_1 must not belong to the grid G*")
        m0, C0 = self.posterior_given_D(np.vstack([v1, self.G]))
        C0 = 0.5 * (C0 + C0.T)
        self.m01, self.C011, self.m0G, self.C0G1, self.C0GG = m0[0], C0[0, 0], m0[1:], C0[1:, 0], C0[1:, 1:]
        self.coef = self.C0G1 / self.C011
        CG = self.C0GG - np.outer(self.C0G1, self.C0G1) / self.C011
        self.LG = np.linalg.cholesky(0.5 * (CG + CG.T) + 1e-8 * np.eye(self.N))   # Step-2 conditional, fixed because v_1 is known

    def coeffs(self, s):
        """(c, B, r): conditional mean of eta(v) given (D,D*) is c + B.D*, residual variance r. s = k(v, P) rows."""
        M = cho_solve(self.cho, s.T).T if self.solver == "chol" else s @ self.Ainv
        return M[:, :self.n] @ self.yd, M[:, self.n:], self.sf2 - (M * s).sum(1)

    def simulate(self, R, T, rng, input_fn, y0=0.0, time_steps=False):
        v1 = input_fn(1, np.full(R, y0))
        self.prepare(v1[0])
        x = np.full((R, T), np.nan)
        first_fail = np.full(R, np.inf)
        g1 = self.m01 + np.sqrt(self.C011) * rng.standard_normal(R)
        Dstar = self.m0G[None, :] + (g1 - self.m01)[:, None] * self.coef[None, :] + rng.standard_normal((R, self.N)) @ self.LG.T
        vals = np.concatenate([np.tile(self.yd, (R, 1)), Dstar], 1)
        W = cho_solve(self.cho, vals.T).T if self.solver == "chol" else vals @ self.Ainv
        x[:, 0] = g1 + (np.sqrt(self.nug) * rng.standard_normal(R) if self.nug else 0.0)
        times = []
        for t in range(2, T + 1):
            t0 = time.perf_counter()
            alive = np.isinf(first_fail)
            v = input_fn(t, np.where(alive, x[:, t - 2], 0.0))
            s = self.k(v, self.P)
            mean = (s * W).sum(1)
            if self.solver == "chol":
                var = self.sf2 - (solve_triangular(self.L, s.T, lower=True, check_finite=False) ** 2).sum(0)
            else:
                var = self.sf2 - ((s @ self.Ainv) * s).sum(1)
            bad = alive & (~np.isfinite(mean) | ~np.isfinite(var) | (var < -1e-8))
            first_fail[bad] = t
            xt = mean + np.sqrt(np.maximum(var, 0.0) + self.nug) * rng.standard_normal(R)
            bad = alive & ~bad & (~np.isfinite(xt) | (np.abs(xt) > 10))
            first_fail[bad] = t
            x[:, t - 1] = np.where(np.isinf(first_fail), xt, np.nan)
            times.append(time.perf_counter() - t0)
        return (x, first_fail, np.array(times)) if time_steps else (x, first_fail)

    # ---- closed-form joints for FIXED inputs V (T,d); v_1 = V[0] ----
    def joint_exact(self, V):
        return self.posterior_given_D(np.atleast_2d(V))

    def joint_lut(self, V):
        V = np.atleast_2d(V)
        self.prepare(V[0])
        m0 = np.concatenate([[self.m01], self.m0G])
        _, C0 = self.posterior_given_D(np.vstack([V[:1], self.G])); C0 = 0.5 * (C0 + C0.T)
        S = self.k(V[1:], self.P)
        M = cho_solve(self.cho, S.T).T
        c, B, r = M[:, :self.n] @ self.yd, M[:, self.n:], self.sf2 - (M * S).sum(1)
        T = len(V)
        mean = np.concatenate([[self.m01], c + B @ self.m0G])
        C = np.zeros((T, T))
        C[0, 0] = C0[0, 0]
        C[1:, 1:] = B @ self.C0GG @ B.T + np.diag(np.maximum(r, 0.0))
        C[0, 1:] = C[1:, 0] = C0[0, 1:] @ B.T
        return mean, C


def exact_chol_trajectory(Xd, yd, kernel, y0, T, input_fn, rng, jitter, nugget=0.0, sf2=1.0):
    """Exact sequential conditioning (marginalised D*): Cholesky append, O(t^2) per step. Returns (x (T,), first_fail)."""
    Xd, yd = np.atleast_2d(Xd), np.asarray(yd, float)
    n, d = len(Xd), Xd.shape[1]
    pts, L, z = np.empty((n + T, d)), np.zeros((n + T, n + T)), np.empty(n + T)
    pts[:n] = Xd
    L[:n, :n] = np.linalg.cholesky(kernel(Xd, Xd) + jitter * np.eye(n) + 1e-300)
    z[:n] = solve_triangular(L[:n, :n], yd, lower=True)
    x = np.full(T, np.nan)
    ylast = y0
    for t in range(1, T + 1):
        m = n + t - 1
        v = input_fn(t, np.array([ylast]))[0]
        kv = kernel(v[None, :], pts[:m])[0]
        l = solve_triangular(L[:m, :m], kv, lower=True, check_finite=False)
        dpiv = sf2 + nugget + jitter - l @ l
        if not np.isfinite(dpiv) or dpiv <= 0:
            return x, float(t)
        mean = l @ z[:m]
        xt = mean + np.sqrt(dpiv) * rng.standard_normal()
        if not np.isfinite(xt) or abs(xt) > 10:
            return x, float(t)
        sq = np.sqrt(dpiv)
        pts[m], L[m, :m], L[m, m], z[m] = v, l, sq, (xt - l @ z[:m]) / sq
        x[t - 1], ylast = xt, xt
    return x, np.inf


def exact_naive_trajectory(Xd, yd, kernel, y0, T, input_fn, rng, nugget=0.0, sf2=1.0):
    """Exact sequential conditioning with a NEW explicit matrix inverse every step (the paper's marginalised version), jitter 0."""
    Xd, yd = np.atleast_2d(Xd), np.asarray(yd, float)
    n, d = len(Xd), Xd.shape[1]
    pts, vals = np.empty((n + T, d)), np.empty(n + T)
    pts[:n], vals[:n] = Xd, yd
    x = np.full(T, np.nan)
    ylast = y0
    for t in range(1, T + 1):
        m = n + t - 1
        v = input_fn(t, np.array([ylast]))[0]
        A = kernel(pts[:m], pts[:m])
        A[n:, n:] += nugget * np.eye(t - 1)
        try:
            Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return x, float(t)
        kv = kernel(v[None, :], pts[:m])[0]
        var = sf2 + nugget - kv @ Ainv @ kv
        mean = kv @ Ainv @ vals[:m]
        if not np.isfinite(var) or var <= 0 or not np.isfinite(mean):
            return x, float(t)
        xt = mean + np.sqrt(var) * rng.standard_normal()
        if not np.isfinite(xt) or abs(xt) > 10:
            return x, float(t)
        pts[m], vals[m] = v, xt
        x[t - 1], ylast = xt, xt
    return x, np.inf
