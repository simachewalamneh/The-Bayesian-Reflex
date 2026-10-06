"""E5-B - Functional drift with the PC-FSVI STAND-IN vs an exact functional Kalman oracle (grid state-space GP).

Scope: validates tracking of a drifting FUNCTION under a known Gaussian state-space model with the stand-in
(not the real pc-fsvi-continual-learning engine; swap-in is the next step). B2/B3 are misspecified-drift trade-off studies.
Oracle : f_t on grid G=41 over [-3,3], prior N(0,K_RBF), f_t=f_{t-1}+eta_t, eta~N(0,seta^2 K) (kernel-correlated drift),
         y_t=f_t(x_t)+N(0,0.2^2), x_t on the grid. Exact Kalman filter (O(G^2) per step).
Stand-in: whitened inducing posterior (M=20), time update S += seta^2 I (or S /= lambda), PC-settling fixed point
         (vectorised closed form here; verified equal to the PCFSVI class update below).
B1  random-walk function (model correct), T=200, seta=0.05, 500 trials x 10 seeds.   PASS/FAIL
B2  global switch sin -> cos at t=100 (T=200).   B3 regional change: f=sin(x) -> sin(x)+smoothstep(x on [0,1]) for t>=100,
    observations only at x>=0 after the change. Tracking region x>=1, retention region x<=-1.
Criteria (fixed before running):
  B1 oracle: cov95 of f_t(grid) within 3 SE (n_eff=5000 trials) at t in {10,50,100,200}; innovation z mean/var/lag-1 within 3 SE.
  B1 stand-in vs oracle (t=50,200): RMSE(mu) < 0.02, relative sd RMSE < 0.10;  stand-in class == vectorised form (|dm|<1e-6).
  B1 CONTROLS (must fail): static (seta=0) cov95(200) < 0.90; seta x0.2 and x5: |cov95(200)-0.95| > 0.03; stand-in M=3 rel sd > 0.10.
  B2: oracle seta=0.1 recovers (median <= 80 steps, >=90% within 100); static (seta=0) <10% recover (must fail).
      recovery = first step after t=100 where grid RMSE <= 1.5 x the method's own mean RMSE over t=90..99, held 5 steps.
  B3: static tracking RMSE (x>=1, t=200) > 2 x oracle seta=0.1 tracking RMSE; oracle seta=0.1 retention ratio
      RMSE_ret(200)/RMSE_ret(99) < 1.5.   Other B2/B3 numbers are informational trade-off curves.
Run: python experiments/e5b_functional_drift.py
"""
import json
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bayesian_reflex.models import PCFSVI  # noqa: E402
from bayesian_reflex.models.function_space import rbf  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
G, SY, T, SETA, N1, N2, SEEDS, TS = 41, 0.2, 200, 0.05, 500, 100, list(range(10)), 100
GRID = np.linspace(-3, 3, G)
K = rbf(GRID, GRID) + 1e-8 * np.eye(G)
LK = np.linalg.cholesky(K)
CPS = [10, 50, 100, 200]


class GridKalman:
    """Exact functional Kalman filter on the grid; P shared across trials (inputs shared), means per trial."""
    def __init__(self, seta, n, lam=None):
        self.m, self.P, self.Q, self.lam = np.zeros((n, G)), K.copy(), seta ** 2 * K, lam

    def step(self, i, y):
        P = self.P / self.lam if self.lam else self.P + self.Q
        s = P[i, i] + SY ** 2
        z = (y - self.m[:, i]) / np.sqrt(s)                      # standardized innovation (BEFORE update)
        g = P[:, i] / s
        self.m = self.m + (y - self.m[:, i])[:, None] * g[None, :]
        self.P = P - np.outer(g, P[i, :])
        return z

    def band(self):
        return self.m, np.sqrt(np.maximum(np.diag(self.P), 1e-12))


class StandIn:
    """Whitened inducing posterior; closed-form fixed point of the PC settling (checked against PCFSVI)."""
    def __init__(self, seta, n, M=20, lam=None):
        self.fs = PCFSVI(M=M, sy=SY)
        self.M, self.seta, self.lam = M, seta, lam
        self.m, self.S = np.zeros((n, M)), np.eye(M)
        self.PhiG = self.fs.phi(GRID)                            # (M, G)

    def step(self, i, y):
        if self.seta:
            self.S = self.S + self.seta ** 2 * np.eye(self.M)
        if self.lam:
            self.S = self.S / self.lam
        phi = self.PhiG[:, i]
        Sphi = self.S @ phi
        den = SY ** 2 + phi @ Sphi
        self.m = self.m + (y - self.m @ phi)[:, None] * (Sphi / den)[None, :]
        self.S = self.S - np.outer(Sphi, Sphi) / den

    def band(self):
        var = 1.0 - (self.PhiG ** 2).sum(0) + (self.PhiG * (self.S @ self.PhiG)).sum(0)
        return self.m @ self.PhiG, np.sqrt(np.maximum(var, 1e-12))


def smoothstep(x):
    u = np.clip(x, 0, 1)
    return u * u * (3 - 2 * u)


def truth(stream, t):                                            # t is 1-based
    if stream == "b2":
        return np.sin(GRID) if t < TS else np.cos(GRID)
    return np.sin(GRID) + (smoothstep(GRID) if t >= TS else 0.0)


def b1_seed(seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, G, T)
    f = (LK @ rng.standard_normal((G, N1))).T
    models = {"oracle": GridKalman(SETA, N1), "static": GridKalman(0.0, N1), "eta_x0.2": GridKalman(0.2 * SETA, N1),
              "eta_x5": GridKalman(5 * SETA, N1), "standin": StandIn(SETA, N1), "standin_M3": StandIn(SETA, N1, M=3)}
    out = {k: {} for k in models}; z = np.empty((N1, T)); truths = {}
    for t in range(T):
        f = f + SETA * (LK @ rng.standard_normal((G, N1))).T
        y = f[:, idx[t]] + SY * rng.standard_normal(N1)
        for k, m in models.items():
            r = m.step(idx[t], y)
            if k == "oracle":
                z[:, t] = r
        if t + 1 in CPS:
            truths[t + 1] = f.copy()
            for k, m in models.items():
                out[k][t + 1] = tuple(a.copy() for a in m.band())
    return out, z, truths


def b23_seed(stream, seed, specs):
    rng = np.random.default_rng(seed)
    hi = G if stream == "b2" else None
    idx = np.array([rng.integers(0, G) if (stream == "b2" or t + 1 < TS) else rng.integers(G // 2, G) for t in range(T)])
    models = {k: (GridKalman(v, N2) if k.startswith("eta") else StandIn(0.0, N2, lam=v)) for k, v in specs.items()}
    rm = {k: np.empty((N2, T)) for k in models}
    reg = {k: {r: np.empty((N2, T)) for r in ("trk", "ret")} for k in models}
    for t in range(T):
        f = truth(stream, t + 1)
        y = f[idx[t]] + SY * rng.standard_normal(N2)
        for k, m in models.items():
            m.step(idx[t], y)
            mu = m.band()[0]
            rm[k][:, t] = np.sqrt(np.mean((mu - f) ** 2, 1))
            reg[k]["trk"][:, t] = np.sqrt(np.mean((mu[:, GRID >= 1] - f[GRID >= 1]) ** 2, 1))
            reg[k]["ret"][:, t] = np.sqrt(np.mean((mu[:, GRID <= -1] - f[GRID <= -1]) ** 2, 1))
    return rm, reg


def recovery(r):
    pre = r[:, 89:99].mean()
    ok = r[:, TS - 1:] <= 1.5 * pre
    sus = ok[:, :-4] & ok[:, 1:-3] & ok[:, 2:-2] & ok[:, 3:-1] & ok[:, 4:]
    first = np.where(sus.any(1), sus.argmax(1), np.inf)
    return float(np.median(first)), float(np.mean(np.isfinite(first) & (first <= 100)))


def main():
    rows = []
    add = lambda n, v, c, p, info=False: rows.append((n + (" [info]" if info else ""), float(v), c, bool(p), info))  # noqa: E731
    # ---------- B1 ----------
    res = [b1_seed(s) for s in SEEDS]
    nt = N1 * len(SEEDS)
    z = np.concatenate([r[1] for r in res]); zn = z.size
    for t in CPS:
        mu = np.concatenate([r[0]["oracle"][t][0] for r in res]); sd = res[0][0]["oracle"][t][1]
        # sd differs per seed (inputs differ): recompute per seed
        cov = np.mean(np.concatenate([np.abs(r[2][t] - r[0]["oracle"][t][0]) <= 1.96 * r[0]["oracle"][t][1] for r in res]))
        tol = 3 * np.sqrt(0.95 * 0.05 / nt)
        add(f"B1 oracle cov95 t={t}", cov, f"|v-0.95|<{tol:.4f}", abs(cov - 0.95) < tol)
    r1 = np.corrcoef(z[:, :-1].ravel(), z[:, 1:].ravel())[0, 1]
    add("B1 oracle innovation mean", z.mean(), f"|v|<{3/np.sqrt(zn):.4f}", abs(z.mean()) < 3 / np.sqrt(zn))
    add("B1 oracle innovation var", z.var(), f"|v-1|<{3*np.sqrt(2/zn):.4f}", abs(z.var() - 1) < 3 * np.sqrt(2 / zn))
    add("B1 oracle innovation lag-1 autocorr", r1, f"|v|<{3/np.sqrt(zn):.4f}", abs(r1) < 3 / np.sqrt(zn))

    def agree(k, t):
        a = np.sqrt(np.mean(np.concatenate([(r[0][k][t][0] - r[0]["oracle"][t][0]) ** 2 for r in res])))
        s = np.sqrt(np.mean([np.mean((r[0][k][t][1] - r[0]["oracle"][t][1]) ** 2) for r in res]))
        s0 = np.sqrt(np.mean([np.mean(r[0]["oracle"][t][1] ** 2) for r in res]))
        return a, s / s0
    for t in (50, 200):
        a, rs = agree("standin", t)
        add(f"B1 stand-in RMSE(mu - oracle) t={t}", a, "<0.02", a < 0.02)
        add(f"B1 stand-in relative sd RMSE t={t}", rs, "<0.10", rs < 0.10)
    cv = lambda k: float(np.mean(np.concatenate([np.abs(r[2][200] - r[0][k][200][0]) <= 1.96 * r[0][k][200][1] for r in res])))  # noqa: E731
    add("CONTROL B1 static cov95(200)", cv("static"), "<0.90 (must fail)", cv("static") < 0.90)
    for k in ("eta_x0.2", "eta_x5"):
        add(f"CONTROL B1 {k} |cov95(200)-0.95|", abs(cv(k) - 0.95), ">0.03 (must fail)", abs(cv(k) - 0.95) > 0.03)
    c3 = agree("standin_M3", 200)[1]
    add("CONTROL B1 stand-in M=3 relative sd RMSE", c3, ">0.10 (must fail)", c3 > 0.10)
    # class-vs-vectorised check (single trial, same data)
    rng = np.random.default_rng(5); idx = rng.integers(0, G, T); fcur = LK @ rng.standard_normal(G); si = StandIn(SETA, 1); cls = PCFSVI(M=20, sy=SY, sigma_eta=SETA)
    for t in range(T):
        fcur = fcur + SETA * LK @ rng.standard_normal(G); y = fcur[idx[t]] + SY * rng.standard_normal()
        si.step(idx[t], np.array([y])); cls.update(y, GRID[idx[t]])
    dm = np.abs(si.m[0] - cls.m).max()
    add("B1 PCFSVI class == vectorised stand-in (max|dm|)", dm, "<1e-6", dm < 1e-6)

    # ---------- B2 / B3 ----------
    specs = {"eta_0 (static)": 0.0, "eta_0.02": 0.02, "eta_0.05": 0.05, "eta_0.1": 0.1, "eta_0.2": 0.2,
             "lam_1.0": 1.0, "lam_0.99": 0.99, "lam_0.98": 0.98, "lam_0.95": 0.95, "lam_0.9": 0.9}
    specs2 = {("eta" if k.startswith("eta") else "lam") + "|" + k: v for k, v in specs.items()}
    # B2/B3 model dispatch: names starting with 'eta' -> GridKalman(seta), else StandIn(lambda)
    specs_run = {k.split("|")[1]: v for k, v in specs2.items()}
    b2 = [b23_seed("b2", s, specs_run) for s in SEEDS[:5]]
    b3 = [b23_seed("b3", s, specs_run) for s in SEEDS[:5]]
    tab = {}
    print("\nB2 global switch sin->cos at t=100 | B3 regional change (tracking x>=1, retention x<=-1)")
    print(f"{'method':16s}{'B2 recov(med)':>14s}{'frac':>6s}{'B2 RMSE@200':>12s}{'B3 trk@200':>11s}{'B3 ret@99':>10s}{'B3 ret@200':>11s}{'ratio':>7s}")
    for k in specs_run:
        r2 = np.concatenate([r[0][k] for r in b2]); med, fr = recovery(r2)
        trk = np.mean(np.concatenate([r[1][k]["trk"][:, -1] for r in b3])); r99 = np.mean(np.concatenate([r[1][k]["ret"][:, 98] for r in b3]))
        r200 = np.mean(np.concatenate([r[1][k]["ret"][:, -1] for r in b3]))
        tab[k] = dict(recov_med=med, recov_frac=fr, b2_rmse200=float(r2[:, -1].mean()), b3_trk200=float(trk), b3_ret99=float(r99), b3_ret200=float(r200), ratio=float(r200 / r99))
        print(f"{k:16s}{med:14.1f}{fr:6.2f}{r2[:, -1].mean():12.3f}{trk:11.3f}{r99:10.3f}{r200:11.3f}{r200 / r99:7.2f}")
    o, st = tab["eta_0.1"], tab["eta_0 (static)"]
    add("B2 oracle seta=0.1 median recovery (steps)", o["recov_med"], "<=80", o["recov_med"] <= 80)
    add("B2 oracle seta=0.1 fraction recovered <=100", o["recov_frac"], ">=0.90", o["recov_frac"] >= 0.90)
    add("CONTROL B2 static fraction recovered", st["recov_frac"], "<0.10 (must fail)", st["recov_frac"] < 0.10)
    add("B3 static tracking RMSE / oracle(0.1) tracking RMSE", st["b3_trk200"] / o["b3_trk200"], ">2", st["b3_trk200"] / o["b3_trk200"] > 2)
    add("B3 oracle(0.1) retention ratio RMSE(200)/RMSE(99)", o["ratio"], "<1.5", o["ratio"] < 1.5)
    for k in ("lam_0.98", "lam_0.95", "lam_0.9"):
        add(f"B2 stand-in {k} median recovery", tab[k]["recov_med"], "info", True, True)

    print(f"\n{'check':58s}{'value':>11s}  criterion")
    for n_, v, c, p, info in rows:
        print(f"{n_:58s}{v:11.3e}  {c:30s}{('PASS' if p else 'FAIL') + (' (info)' if info else '')}")
    ok = all(r[3] for r in rows if not r[4])
    print(f"\nchecks: {sum(r[3] for r in rows if not r[4])}/{sum(1 for r in rows if not r[4])} pass")
    print("E5-B (functional drift, PC-FSVI STAND-IN vs exact functional Kalman oracle):", "PASS" if ok else "FAIL")

    # ---------- figures ----------
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    r0, f0 = res[0][0], res[0][2][200][0]
    a = ax[0, 0]; a.plot(GRID, f0, "k--", label="true f_200")
    for k, col, nm in (("oracle", "tab:blue", "exact functional Kalman"), ("standin", "tab:red", "PC-FSVI stand-in"), ("static", "tab:green", "static (no drift)")):
        mu, sd = r0[k][200][0][0], r0[k][200][1]
        a.plot(GRID, mu, color=col, label=nm); a.fill_between(GRID, mu - 2 * sd, mu + 2 * sd, color=col, alpha=0.15)
    a.set_title("B1 random-walk function, one trial, t=200"); a.legend(fontsize=7)
    tt = np.arange(1, T + 1)
    for k, col in (("eta_0 (static)", "tab:green"), ("eta_0.05", "tab:blue"), ("eta_0.1", "tab:purple"), ("lam_0.95", "tab:red"), ("lam_0.9", "tab:orange")):
        ax[0, 1].plot(tt, np.concatenate([r[0][k] for r in b2]).mean(0), color=col, label=k)
    ax[0, 1].axvline(TS, color="k", ls=":"); ax[0, 1].set_yscale("log"); ax[0, 1].set_title("B2 grid RMSE after global switch"); ax[0, 1].legend(fontsize=7)
    names = list(specs_run)
    ax[1, 0].bar(np.arange(len(names)) - 0.2, [tab[k]["b3_trk200"] for k in names], 0.4, label="tracking RMSE (x>=1)")
    ax[1, 0].bar(np.arange(len(names)) + 0.2, [tab[k]["b3_ret200"] for k in names], 0.4, label="retention RMSE (x<=-1)")
    ax[1, 0].set_xticks(range(len(names))); ax[1, 0].set_xticklabels(names, rotation=60, fontsize=7); ax[1, 0].set_title("B3 regional change, t=200"); ax[1, 0].legend(fontsize=7)
    ax[1, 1].plot([tab[k]["b3_trk200"] for k in names], [tab[k]["b3_ret200"] for k in names], "o")
    for k in names:
        ax[1, 1].annotate(k, (tab[k]["b3_trk200"], tab[k]["b3_ret200"]), fontsize=6)
    ax[1, 1].set_xlabel("tracking RMSE (x>=1)"); ax[1, 1].set_ylabel("retention RMSE (x<=-1)"); ax[1, 1].set_title("B3 tracking vs retention")
    fig.tight_layout(); fig.savefig(OUT / "e5b_functional_drift_comparison.png", dpi=140); plt.close(fig)
    (OUT / "e5b_results.json").write_text(json.dumps(dict(
        scope="functional drift, PC-FSVI stand-in vs exact grid Kalman oracle; B2/B3 are misspecified trade-off studies; real PC-FSVI not yet used",
        overall_pass=ok, checks=[dict(name=n_, value=v, criterion=c, passed=p, informational=i) for n_, v, c, p, i in rows], tradeoff=tab), indent=2))


if __name__ == "__main__":
    main()
