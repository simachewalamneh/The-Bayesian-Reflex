"""E2a - Sequential updating vs batch oracle (scalar Gaussian module).

Scope: validates SEQUENTIAL UPDATING of the scalar Gaussian inference module (paper Thm 2.1,
conjugate case). It does not test prediction error, function space, action or the closed loop.

Question : Does the recursive update match the batch posterior over a long horizon, and stay numerically stable?
Oracle   : batch posterior from y_{1:t} in closed form, computed in extended precision (longdouble),
           so it is independent of the float64 recursion being tested.
Primary  : T=100, 100 random problems (random mu0, s0, sy, theta). Compare mu_t, sigma_t^2, KL(seq||batch) at every t.
Criteria : max|dmu| < 1e-6, max|dvar| < 1e-6, max KL < 1e-12 (all t, all problems).
Stability: T=100000 (20 problems): same criteria, finite values, error growth reported.
Extra    : permutation invariance (<1e-10); NEGATIVE CONTROL: dropping every 10th observation must change mu (>1e-3).
Run: python experiments/e2_sequential_equivalence.py
"""
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bayesian_reflex.models import ScalarGaussianBelief  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
LD = np.longdouble


def make_problems(rng, n, T):
    mu0 = rng.uniform(-3, 3, n)
    s0 = rng.uniform(0.3, 3.0, n)
    sy = rng.uniform(0.1, 2.0, n)
    theta = mu0 + s0 * rng.standard_normal(n)
    y = theta[:, None] + sy[:, None] * rng.standard_normal((n, T))
    return mu0, s0, sy, y


def sequential(mu0, s0, sy, y):
    n, T = y.shape
    b = ScalarGaussianBelief(1.0, 1.0, 1.0, n)                      # overwrite per-problem params
    b.mu, b.prec, b.sy2 = mu0.copy(), 1.0 / s0 ** 2, sy ** 2
    mu, var = np.empty((n, T)), np.empty((n, T))
    for t in range(T):
        b.update(y[:, t])
        mu[:, t], var[:, t] = b.mu, b.var
    return mu, var


def batch_oracle(mu0, s0, sy, y):
    """Posterior from y_{1:t}: precision = 1/s0^2 + t/sy^2; mean = (mu0/s0^2 + sum y / sy^2)/precision."""
    mu0, s0, sy, y = (a.astype(LD) for a in (mu0, s0, sy, y))
    t = np.arange(1, y.shape[1] + 1, dtype=LD)[None, :]
    prec = 1 / s0[:, None] ** 2 + t / sy[:, None] ** 2
    mean = (mu0[:, None] / s0[:, None] ** 2 + np.cumsum(y, axis=1) / sy[:, None] ** 2) / prec
    return mean, 1 / prec


def kl_gauss(mu_s, var_s, mu_b, var_b):
    """KL(N(mu_s,var_s) || N(mu_b,var_b)); log1p form avoids cancellation when the two are nearly equal."""
    d = (var_s - var_b) / var_b
    return 0.5 * (d - np.log1p(d)) + (mu_s - mu_b) ** 2 / (2 * var_b)


def compare(seq, orc):
    (mu_s, var_s), (mu_b, var_b) = seq, orc
    dmu = np.abs(mu_s.astype(LD) - mu_b)
    dvar = np.abs(var_s.astype(LD) - var_b)
    kl = kl_gauss(mu_s.astype(LD), var_s.astype(LD), mu_b, var_b)
    return dmu.astype(float), dvar.astype(float), (dvar / var_b).astype(float), kl.astype(float)


def main():
    rng = np.random.default_rng(0)
    rows = []

    # ---- primary: T = 100 ----
    T = 100
    mu0, s0, sy, y = make_problems(rng, 100, T)
    seq, orc = sequential(mu0, s0, sy, y), batch_oracle(mu0, s0, sy, y)
    dmu, dvar, rvar, kl = compare(seq, orc)
    rows += [("T=100 max|dmu|", dmu.max(), "<1e-6", dmu.max() < 1e-6),
             ("T=100 max|dvar|", dvar.max(), "<1e-6", dvar.max() < 1e-6),
             ("T=100 max KL", kl.max(), "<1e-12", kl.max() < 1e-12)]
    diag = dict(max_rel_dvar_T100=float(rvar.max()))

    # ---- permutation invariance ----
    perm = rng.permutation(T)
    mu_p, var_p = sequential(mu0, s0, sy, y[:, perm])
    pi = float(np.abs(mu_p[:, -1] - seq[0][:, -1]).max())
    rows.append(("permutation |dmu| (final)", pi, "<1e-10", pi < 1e-10))

    # ---- negative control: drop every 10th observation ----
    keep = np.array([i for i in range(T) if (i + 1) % 10 != 0])
    mu_d, _ = sequential(mu0, s0, sy, y[:, keep])
    nc = float(np.abs(mu_d[:, -1].astype(LD) - orc[0][:, -1]).astype(float).mean())
    rows.append(("CONTROL mean|dmu| (dropped data)", nc, ">1e-3 (must differ)", nc > 1e-3))

    # ---- stability stress: T = 100000 ----
    Ts = 100_000
    mu0s, s0s, sys_, ys = make_problems(rng, 20, Ts)
    seq_s, orc_s = sequential(mu0s, s0s, sys_, ys), batch_oracle(mu0s, s0s, sys_, ys)
    dmu_s, dvar_s, rvar_s, kl_s = compare(seq_s, orc_s)
    finite = bool(np.isfinite(seq_s[0]).all() and np.isfinite(seq_s[1]).all())
    rows += [("T=1e5 finite values", float(finite), "True", finite),
             ("T=1e5 max|dmu|", dmu_s.max(), "<1e-6", dmu_s.max() < 1e-6),
             ("T=1e5 max rel dvar", rvar_s.max(), "<1e-6", rvar_s.max() < 1e-6),
             ("T=1e5 max KL", kl_s.max(), "<1e-12", kl_s.max() < 1e-12)]
    t_ax = np.arange(1, Ts + 1)
    worst = dmu_s.max(0)
    sel = t_ax >= 10
    diag["dmu_growth_loglog_slope_T1e5"] = float(np.polyfit(np.log(t_ax[sel]), np.log(np.maximum(worst[sel], 1e-300)), 1)[0])

    print(f"{'check':36s}{'value':>12s}  criterion")
    for n, v, c, p in rows:
        print(f"{n:36s}{v:12.3e}  {c:22s}{'PASS' if p else 'FAIL'}")
    print("diagnostics:", diag)
    ok = all(r[3] for r in rows)
    print("\nE2a (scalar Gaussian sequential updating vs batch oracle):", "PASS" if ok else "FAIL")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].semilogy(np.arange(1, T + 1), np.maximum(dmu.max(0), 1e-20), label="max |dmu|")
    ax[0].semilogy(np.arange(1, T + 1), np.maximum(dvar.max(0), 1e-20), label="max |dvar|")
    ax[0].semilogy(np.arange(1, T + 1), np.maximum(kl.max(0), 1e-20), label="max KL")
    ax[0].axhline(1e-6, color="k", ls=":", label="1e-6 criterion"); ax[0].set_title("T=100 (100 problems)")
    ax[1].loglog(t_ax, np.maximum(worst, 1e-20), label="max |dmu|")
    ax[1].loglog(t_ax, np.maximum(dvar_s.max(0), 1e-20), label="max |dvar|")
    ax[1].loglog(t_ax, np.maximum(kl_s.max(0), 1e-20), label="max KL")
    ax[1].axhline(1e-6, color="k", ls=":"); ax[1].set_title("stability stress T=1e5 (20 problems)")
    for a in ax:
        a.set_xlabel("observations t"); a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "e2_sequential_vs_batch.png", dpi=140); plt.close(fig)

    (OUT / "e2_results.json").write_text(json.dumps(dict(
        scope="scalar Gaussian sequential updating vs batch oracle (longdouble)", overall_pass=bool(ok),
        checks=[dict(name=n, value=float(v), criterion=c, passed=bool(p)) for n, v, c, p in rows],
        diagnostics=diag), indent=2))


if __name__ == "__main__":
    main()
