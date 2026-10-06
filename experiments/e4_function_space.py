"""E4 - Function-space belief: exact GP (oracle) vs PC-FSVI stand-in.

Scope: compares a function-space variational stand-in (inducing-point q(u), predictive-coding settling update;
see src/bayesian_reflex/models/function_space.py) with the exact GP oracle on a streaming sin(x) task. It is NOT
a test of the pc-fsvi-continual-learning repo code, and says nothing about nonstationarity, action or the full loop.

Task    : f=sin(x), x_t ~ U[-3,3], y_t=f(x_t)+N(0,0.2^2), T=1000, test grid 200 pts, 10 seeds.
Shared  : RBF prior sf=1, ell=1, known sigma_y=0.2 (hyperparameters fixed so the comparison isolates inference).
Metrics : MSE vs truth; RMSE(mu_fs - mu_gp); relative RMSE of epistemic sd (sd of f) vs oracle; test LL on fresh noisy
          labels; 95% interval coverage; latency (batch refit O(t^3), rank-one GP O(t^2), PC-FSVI O(M^2)).
Criteria (fixed before running):
  E2b  rank-one sequential GP == batch GP: max|dmu|,|dsd| < 1e-8 at all checkpoints.
  PC   settling reaches the conjugate fixed point (T=1000, M=20): max|dm|,|dS| < 1e-6;  CONTROL K=2 iterations: >1e-4.
  Agree (M=20, T in {50,1000}): RMSE(mu) < 0.01, relative sd RMSE < 0.10, |LL gap| < 0.02 nats, MSE within 5% of GP.
  Sanity GP MSE(T=1000) < MSE(T=10).  CONTROL M=3 inducing points: relative sd RMSE at T=1000 > 0.10.
  Latency: PC-FSVI time(t=1000)/time(t=50) < 3;  batch-refit time ratio > 20 (indicative, machine dependent).
Run: python experiments/e4_function_space.py
"""
import json
import pathlib
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import cho_factor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bayesian_reflex.models import ExactGP, PCFSVI, batch_gp  # noqa: E402
from bayesian_reflex.models.function_space import rbf  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SY, T, SEEDS = 0.2, 1000, list(range(10))
CPS = [10, 50, 200, 1000]
XS = np.linspace(-3, 3, 200)
FS = np.sin(XS)
SPEC = {"M3": (3, 30), "M5": (5, 30), "M10": (10, 30), "M20": (20, 30), "M40": (40, 30), "M20_K2": (20, 2)}
MAIN = "M20"


def metrics(mu, var, mu_g, var_g, ytest):
    sd, sd_g = np.sqrt(var), np.sqrt(var_g)
    s = np.sqrt(var + SY ** 2)
    return dict(mse=np.mean((mu - FS) ** 2), agree_mu=np.sqrt(np.mean((mu - mu_g) ** 2)),
                rel_sd=np.sqrt(np.mean((sd - sd_g) ** 2)) / np.sqrt(np.mean(sd_g ** 2)),
                ll=np.mean(stats.norm.logpdf(ytest, mu, s)), cov=np.mean(np.abs(ytest - mu) < 1.96 * s))


def run_seed(seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, T)
    y = np.sin(x) + SY * rng.standard_normal(T)
    ytest = FS + SY * rng.standard_normal(len(XS))
    gp = ExactGP(sy=SY, capacity=T)
    fs = {k: PCFSVI(M=M, sy=SY, n_iters=K) for k, (M, K) in SPEC.items()}
    tg, tf = np.empty(T), np.empty(T)
    rec = {}
    for t in range(T):
        t0 = time.perf_counter(); gp.update(y[t], x[t]); tg[t] = time.perf_counter() - t0
        for k, m in fs.items():
            t0 = time.perf_counter(); m.update(y[t], x[t])
            if k == MAIN:
                tf[t] = time.perf_counter() - t0
        tt = t + 1
        if tt in CPS:
            mu_g, var_g = gp.predict_f(XS)
            mu_b, var_b = batch_gp(x[:tt], y[:tt], XS, sy=SY)
            r = dict(gp=metrics(mu_g, var_g, mu_g, var_g, ytest),
                     e2b=(np.abs(mu_g - mu_b).max(), np.abs(np.sqrt(var_g) - np.sqrt(var_b)).max()))
            for k, m in fs.items():
                mu, var = m.predict_f(XS)
                mb, Sb = m.batch_posterior(x[:tt], y[:tt])
                r[k] = metrics(mu, var, mu_g, var_g, ytest)
                r[k].update(dm=np.abs(m.m - mb).max(), dS=np.abs(m.S - Sb).max())
            K = rbf(x[:tt], x[:tt]) + SY ** 2 * np.eye(tt)
            ts = []
            for _ in range(3):
                t0 = time.perf_counter(); cho_factor(K, lower=True); ts.append(time.perf_counter() - t0)
            r["lat"] = dict(batch=np.median(ts), gp=np.median(tg[tt - 5:tt]), fs=np.median(tf[tt - 5:tt]))
            if seed == 0:
                r["plot"] = dict(x=x[:tt].copy(), y=y[:tt].copy(), gp=(mu_g, var_g), fs=fs[MAIN].predict_f(XS))
            rec[tt] = r
    return rec


def avg(recs, cp, model, key):
    return float(np.mean([r[cp][model][key] for r in recs]))


def main():
    recs = [run_seed(s) for s in SEEDS]
    rows = []
    ok = lambda n, v, c, p: rows.append((n, float(v), c, bool(p)))  # noqa: E731

    e2b_mu = max(r[cp]["e2b"][0] for r in recs for cp in CPS)
    e2b_sd = max(r[cp]["e2b"][1] for r in recs for cp in CPS)
    ok("E2b GP seq vs batch max|dmu|", e2b_mu, "<1e-8", e2b_mu < 1e-8)
    ok("E2b GP seq vs batch max|dsd|", e2b_sd, "<1e-8", e2b_sd < 1e-8)
    dm = max(r[1000][MAIN]["dm"] for r in recs); dS = max(r[1000][MAIN]["dS"] for r in recs)
    ok("PC settling fixed point max|dm| (M20,T=1000)", dm, "<1e-6", dm < 1e-6)
    ok("PC settling fixed point max|dS| (M20,T=1000)", dS, "<1e-6", dS < 1e-6)
    dk = avg(recs, 1000, "M20_K2", "dm")
    ok("CONTROL K=2 settling mean|dm|", dk, ">1e-4 (must fail)", dk > 1e-4)
    for cp in (50, 1000):
        a, rs = avg(recs, cp, MAIN, "agree_mu"), avg(recs, cp, MAIN, "rel_sd")
        ll = abs(avg(recs, cp, "gp", "ll") - avg(recs, cp, MAIN, "ll"))
        mr = avg(recs, cp, MAIN, "mse") / avg(recs, cp, "gp", "mse") - 1
        ok(f"T={cp} RMSE(mu_fs - mu_gp)", a, "<0.01", a < 0.01)
        ok(f"T={cp} relative sd RMSE", rs, "<0.10", rs < 0.10)
        ok(f"T={cp} |LL gp - LL fs|", ll, "<0.02", ll < 0.02)
        ok(f"T={cp} MSE excess vs GP", mr, "<0.05", mr < 0.05)
    g10, g1000 = avg(recs, 10, "gp", "mse"), avg(recs, 1000, "gp", "mse")
    ok("GP MSE falls T=10 -> T=1000", g1000, f"<{g10:.4f}", g1000 < g10)
    c3 = avg(recs, 1000, "M3", "rel_sd")
    ok("CONTROL M=3 relative sd RMSE (T=1000)", c3, ">0.10 (must fail)", c3 > 0.10)
    lat = {cp: {k: float(np.median([r[cp]["lat"][k] for r in recs])) for k in ("batch", "gp", "fs")} for cp in CPS}
    rf, rb, rg = (lat[1000][k] / lat[50][k] for k in ("fs", "batch", "gp"))
    ok("latency PC-FSVI t1000/t50", rf, "<3", rf < 3)
    ok("latency batch refit t1000/t50", rb, ">20", rb > 20)

    print(f"{'check':46s}{'value':>11s}  criterion")
    for n, v, c, p in rows:
        print(f"{n:46s}{v:11.3e}  {c:24s}{'PASS' if p else 'FAIL'}")
    print("\nmean over 10 seeds (GP oracle | PC-FSVI M=20):  MSE_truth, RMSE(mu agree), rel sd err, test LL, cov95")
    for cp in CPS:
        g, f = ("gp", MAIN)
        print(f"T={cp:4d}  GP {avg(recs,cp,g,'mse'):.4f} {0:.4f} {0:.3f} {avg(recs,cp,g,'ll'):.3f} {avg(recs,cp,g,'cov'):.3f}"
              f" | FS {avg(recs,cp,f,'mse'):.4f} {avg(recs,cp,f,'agree_mu'):.4f} {avg(recs,cp,f,'rel_sd'):.3f}"
              f" {avg(recs,cp,f,'ll'):.3f} {avg(recs,cp,f,'cov'):.3f}")
    print("M sweep, relative sd error at T=200:", {k: round(avg(recs, 200, k, 'rel_sd'), 3) for k in ("M3", "M5", "M10", "M20", "M40")})
    print("latency (s) at t=" + ", ".join(map(str, CPS)))
    for k, nm in (("batch", "exact batch refit O(t^3)"), ("gp", "exact rank-one O(t^2)"), ("fs", "PC-FSVI O(M^2)")):
        print(f"  {nm:26s}", [f"{lat[cp][k]*1e3:.3f}ms" for cp in CPS])
    allpass = all(r[3] for r in rows)
    print("\nE4 (PC-FSVI stand-in vs exact GP oracle, sin(x) stream):", "PASS" if allpass else "FAIL")

    # ---- figures ----
    pl = recs[0]
    fig, ax = plt.subplots(3, 2, figsize=(11, 10), sharex=True, sharey=True)
    for i, cp in enumerate([10, 50, 200]):
        p = pl[cp]["plot"]
        for j, (nm, (mu, var)) in enumerate((("exact GP (oracle)", p["gp"]), ("PC-FSVI stand-in (M=20)", p["fs"]))):
            a = ax[i, j]; sd = np.sqrt(var)
            a.plot(XS, FS, "k--", lw=1, label="true f")
            a.scatter(p["x"], p["y"], s=6, alpha=0.4, color="gray", label="observations")
            a.plot(XS, mu, color="tab:blue" if j == 0 else "tab:red", label="mean")
            a.fill_between(XS, mu - 2 * sd, mu + 2 * sd, color="tab:blue" if j == 0 else "tab:red", alpha=0.25, label="±2σ (epistemic)")
            a.set_title(f"{nm}, T={cp}", fontsize=9)
            if i == 0 and j == 0:
                a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "e4_function_space_comparison.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax[0, 0].loglog(CPS, [avg(recs, c, MAIN, "agree_mu") for c in CPS], "o-", label="RMSE(mu_fs - mu_gp)")
    ax[0, 0].loglog(CPS, [avg(recs, c, MAIN, "rel_sd") for c in CPS], "s-", label="relative sd RMSE")
    ax[0, 0].set_title("agreement with GP oracle (M=20)"); ax[0, 0].set_xlabel("T"); ax[0, 0].legend(fontsize=7)
    Ms = [3, 5, 10, 20, 40]
    for c in (50, 200, 1000):
        ax[0, 1].semilogy(Ms, [avg(recs, c, f"M{m}", "rel_sd") for m in Ms], "o-", label=f"T={c}")
    ax[0, 1].axhline(0.1, color="k", ls=":"); ax[0, 1].set_xlabel("inducing points M"); ax[0, 1].set_title("relative sd error vs M"); ax[0, 1].legend(fontsize=7)
    for k, nm in (("batch", "exact batch refit O(t³)"), ("gp", "exact rank-one O(t²)"), ("fs", "PC-FSVI O(M²)")):
        ax[1, 0].loglog(CPS, [lat[c][k] for c in CPS], "o-", label=nm)
    ax[1, 0].set_title("update latency (s)"); ax[1, 0].set_xlabel("t"); ax[1, 0].legend(fontsize=7)
    ax[1, 1].loglog(CPS, [avg(recs, c, "gp", "mse") for c in CPS], "o-", label="exact GP")
    ax[1, 1].loglog(CPS, [avg(recs, c, MAIN, "mse") for c in CPS], "s--", label="PC-FSVI M=20")
    ax[1, 1].set_title("MSE vs true f"); ax[1, 1].set_xlabel("T"); ax[1, 1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "e4_metrics.png", dpi=140); plt.close(fig)

    summ = {f"T{cp}": {m: {k: avg(recs, cp, m, k) for k in ("mse", "agree_mu", "rel_sd", "ll", "cov")} for m in ["gp"] + list(SPEC)} for cp in CPS}
    (OUT / "e4_results.json").write_text(json.dumps(dict(
        scope="PC-FSVI stand-in (inducing-point q(u), PC settling) vs exact GP oracle; not the pc-fsvi-continual-learning code",
        overall_pass=allpass, checks=[dict(name=n, value=v, criterion=c, passed=p) for n, v, c, p in rows],
        latency_seconds=lat, summary=summ), indent=2))


if __name__ == "__main__":
    main()
