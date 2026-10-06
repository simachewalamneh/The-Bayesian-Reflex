"""E4-real - the user's REAL PC-FSVI (src/bayesian_reflex/external/pc_fsvi) vs the exact GP oracle.

Scope: same sin(x) stream and metrics as E4, now with the repo's own FSVI + pc_infer through an adapter
(models/pcfsvi_adapter.py). utils.py was not uploaded, so rbf_kernel is a shim (standard RBF assumed).
Streaming = each observation is a 1-point 'task' chained as in pc_fsvi_continual (beta=1 => exact Bayes in the sparse model).
Configs : A  n_iters=150, lr=1.0 (values used in pc_fsvi_continual), chunk 1     [primary]
          C  same, chunk 50 (task-style batches)                                  [primary]
          B/D n_iters=50, lr=0.1 (pc_infer function defaults), chunk 1 / 50       [informational]
          CONTROL A with M=3 inducing points (must fail agreement)
Criteria (fixed before the full run, same as E4 for agreement):
  fixed point vs closed_form_optimum (T=1000): max|dm| < 1e-3, max|dS| < 1e-4
  agreement at T in {50,1000}: RMSE(mu) < 0.01, relative sd RMSE < 0.10, |LL gap| < 0.02, MSE excess < 5%
  latency: real update time t=1000 / t=50 < 3 (constant in t)
Run: python experiments/e4_real_pcfsvi.py   (~3-5 min)
"""
import json
import pathlib
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
import e4_function_space as e4  # noqa: E402  (reuses metrics, constants)
from bayesian_reflex.external.pc_fsvi import closed_form_optimum  # noqa: E402
from bayesian_reflex.models import ExactGP, PCFSVI, batch_gp  # noqa: E402
from bayesian_reflex.models.function_space import rbf  # noqa: E402
from bayesian_reflex.models.pcfsvi_adapter import RealPCFSVI  # noqa: E402

OUT = ROOT / "results"
SY, T, SEEDS, XS, FS = e4.SY, e4.T, e4.SEEDS, e4.XS, e4.FS
CPS = [50, 200, 1000]
SPEC = {"A": (20, 150, 1.0, 1), "B": (20, 50, 0.1, 1), "C": (20, 150, 1.0, 50), "D": (20, 50, 0.1, 50), "A_M3": (3, 150, 1.0, 1)}
PRIMARY = ["A", "C"]


def run_seed(seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, T)
    y = np.sin(x) + SY * rng.standard_normal(T)
    ytest = FS + SY * rng.standard_normal(len(XS))
    gp, standin = ExactGP(sy=SY, capacity=T), PCFSVI(M=20, sy=SY)
    real = {k: RealPCFSVI(M=M, sy=SY, n_iters=n, lr=lr) for k, (M, n, lr, ch) in SPEC.items()}
    tg, ts, tr = np.empty(T), np.empty(T), np.empty(T)
    rec = {}
    for t in range(T):
        t0 = time.perf_counter(); gp.update(y[t], x[t]); tg[t] = time.perf_counter() - t0
        t0 = time.perf_counter(); standin.update(y[t], x[t]); ts[t] = time.perf_counter() - t0
        for k, (M, n, lr, ch) in SPEC.items():
            if ch == 1:
                t0 = time.perf_counter(); real[k].update(y[t], x[t])
                if k == "A":
                    tr[t] = time.perf_counter() - t0
            elif (t + 1) % ch == 0:
                real[k].update_batch(x[t + 1 - ch:t + 1], y[t + 1 - ch:t + 1])
        tt = t + 1
        if tt in CPS:
            mu_g, var_g = gp.predict_f(XS)
            r = dict(gp=e4.metrics(mu_g, var_g, mu_g, var_g, ytest))
            for k, m in real.items():
                mu, var = m.predict_f(XS)
                r[k] = e4.metrics(mu, var, mu_g, var_g, ytest)
                mb, Sb = closed_form_optimum(m.model, x[:tt].reshape(-1, 1), y[:tt], np.zeros(m.model.M), m.model.Kzz, 1.0)
                r[k].update(dm=np.abs(m.m - mb).max(), dS=np.abs(m.S - Sb).max())
            K = rbf(x[:tt], x[:tt]) + SY ** 2 * np.eye(tt)
            tb = []
            for _ in range(3):
                t0 = time.perf_counter(); cho_factor(K, lower=True); tb.append(time.perf_counter() - t0)
            r["lat"] = dict(batch=np.median(tb), gp=np.median(tg[tt - 5:tt]), standin=np.median(ts[tt - 5:tt]), real=np.median(tr[tt - 5:tt]))
            if seed == 0:
                r["plot"] = dict(x=x[:tt].copy(), y=y[:tt].copy(), gp=(mu_g, var_g), real=real["A"].predict_f(XS), standin=standin.predict_f(XS))
            rec[tt] = r
    return rec


def main():
    recs = [run_seed(s) for s in SEEDS]
    avg = lambda cp, m, k: float(np.mean([r[cp][m][k] for r in recs]))  # noqa: E731
    rows = []
    add = lambda n, v, c, p, info=False: rows.append((n + (" [info]" if info else ""), float(v), c, bool(p), info))  # noqa: E731

    for k in PRIMARY + ["B", "D"]:
        info = k not in PRIMARY
        dm, dS = max(r[1000][k]["dm"] for r in recs), max(r[1000][k]["dS"] for r in recs)
        add(f"{k} fixed point max|dm| (T=1000)", dm, "<1e-3", dm < 1e-3, info)
        add(f"{k} fixed point max|dS| (T=1000)", dS, "<1e-4", dS < 1e-4, info)
        for cp in (50, 1000):
            a, rs = avg(cp, k, "agree_mu"), avg(cp, k, "rel_sd")
            ll, mr = abs(avg(cp, "gp", "ll") - avg(cp, k, "ll")), avg(cp, k, "mse") / avg(cp, "gp", "mse") - 1
            add(f"{k} T={cp} RMSE(mu_real - mu_gp)", a, "<0.01", a < 0.01, info)
            add(f"{k} T={cp} relative sd RMSE", rs, "<0.10", rs < 0.10, info)
            add(f"{k} T={cp} |LL gp - LL real|", ll, "<0.02", ll < 0.02, info)
            add(f"{k} T={cp} MSE excess vs GP", mr, "<0.05", mr < 0.05, info)
    c3 = avg(1000, "A_M3", "rel_sd")
    add("CONTROL real M=3 relative sd RMSE (T=1000)", c3, ">0.10 (must fail)", c3 > 0.10)
    lat = {cp: {k: float(np.median([r[cp]["lat"][k] for r in recs])) for k in ("batch", "gp", "standin", "real")} for cp in CPS}
    rr = lat[1000]["real"] / lat[50]["real"]
    add("latency real PC-FSVI t1000/t50", rr, "<3", rr < 3)

    print(f"{'check':52s}{'value':>11s}  criterion")
    for n, v, c, p, info in rows:
        print(f"{n:52s}{v:11.3e}  {c:22s}{('PASS' if p else 'FAIL') + (' (info)' if info else '')}")
    print("\nmean over seeds, T=1000: MSE_truth | RMSE(mu agree) | rel sd err | LL | cov95")
    for k in ["gp", "A", "B", "C", "D"]:
        print(f"  {k:3s} {avg(1000,k,'mse'):.5f} {avg(1000,k,'agree_mu'):.2e} {avg(1000,k,'rel_sd'):.2e} {avg(1000,k,'ll'):.4f} {avg(1000,k,'cov'):.3f}")
    print("latency (ms) at t=" + ", ".join(map(str, CPS)))
    for k, nm in (("batch", "exact batch refit"), ("gp", "exact rank-one"), ("standin", "stand-in PC-FSVI"), ("real", "REAL PC-FSVI (A)")):
        print(f"  {nm:20s}", [f"{lat[cp][k]*1e3:.3f}" for cp in CPS])
    allpass = all(r[3] for r in rows if not r[4])
    print("\nE4-real (user's PC-FSVI code vs exact GP oracle, sin(x) stream; primary configs):", "PASS" if allpass else "FAIL")

    p0 = recs[0]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for i, cp in enumerate([50, 200]):
        p = p0[cp]["plot"]
        for j, (nm, (mu, var), col) in enumerate((("exact GP (oracle)", p["gp"], "tab:blue"), ("REAL PC-FSVI (config A)", p["real"], "tab:red"))):
            a, sd = ax[i, j], np.sqrt(var)
            a.plot(XS, FS, "k--", lw=1, label="true f"); a.scatter(p["x"], p["y"], s=6, alpha=0.4, color="gray")
            a.plot(XS, mu, color=col, label="mean"); a.fill_between(XS, mu - 2 * sd, mu + 2 * sd, color=col, alpha=0.25, label="±2σ (epistemic)")
            a.set_title(f"{nm}, T={cp}", fontsize=9)
            if i == 0 and j == 0:
                a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "e4_real_pcfsvi_comparison.png", dpi=140); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for k, nm in (("batch", "exact batch refit"), ("gp", "exact rank-one"), ("standin", "stand-in PC-FSVI"), ("real", "REAL PC-FSVI (A)")):
        ax[0].loglog(CPS, [lat[c][k] for c in CPS], "o-", label=nm)
    ax[0].set_title("update latency (s)"); ax[0].set_xlabel("t"); ax[0].legend(fontsize=7)
    for k in ["A", "B", "C", "D"]:
        ax[1].loglog(CPS, [avg(c, k, "agree_mu") for c in CPS], "o-", label=f"config {k}")
    ax[1].set_title("RMSE(mu_real - mu_gp)"); ax[1].set_xlabel("T"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "e4_real_pcfsvi_metrics.png", dpi=140); plt.close(fig)

    (OUT / "e4_real_results.json").write_text(json.dumps(dict(
        scope="user's real FSVI+pc_infer (adapter, utils shim) vs exact GP oracle on a sin(x) stream; not the continual-learning claims",
        primary_pass=allpass, checks=[dict(name=n, value=v, criterion=c, passed=p, informational=i) for n, v, c, p, i in rows],
        latency_seconds=lat), indent=2))


if __name__ == "__main__":
    main()
