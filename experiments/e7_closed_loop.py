"""E7 - Closed Bayesian reflex loop with ablations (exact-grid reference agent). Protocol: docs/e7_protocol.md (thresholds FROZEN).

Scope: tests that the integrated loop (belief with drift, standardized prediction error, error-driven plasticity,
uncertainty-driven action) behaves coherently under drift plus an abrupt shift and that each component earns its place.
It does not test the paper's look-up-table / ellipsoidal machinery (E4b/E8 deferred) nor the stand-in/real engines (later confirmation).
Environment: f_0~GP(0,K); f_t=f_{t-1}+eta_t, eta~N(0,0.05^2 K); f += cos x - sin x at t=100; sigma_y=0.2; G=41; T=200; 200 trials x 10 seeds.
Agents: FULL, A1 no action (random queries), A2 no drift model (sigma_eta=0), A3 no PC feedback (no plasticity), A4 none, CONTROL (sigma_y x0.5).
Criteria (frozen): C1 FULL RMSE < A1; C2 FULL < A2 (paired t-test over seeds, p<0.01); C3 median recovery FULL <= 0.8 x A3;
 C4 upper 95% CI of (RMSE_FULL-RMSE_A3)/RMSE_A3 over t=20..99 < +5%; C5 |cov95-0.95|<0.03 (FULL, t=20..99);
 C6 A3: mean z^2 over t=101..105 > 3 and mean z^2 over t=20..99 within 3 SE of 1; CONTROL: z^2 over t=20..99 > 1.5.
Run: python experiments/e7_closed_loop.py            (full pre-registered run)
     python experiments/e7_closed_loop.py --smoke    (1 seed x 10 trials, executes only, no criteria)
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
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import e5b_functional_drift as e5b  # noqa: E402
from bayesian_reflex.models.grid_reflex import GridReflexAgent  # noqa: E402

OUT = ROOT / "results"
G, GRID, K, LK, SY, SETA, T, TS = e5b.G, e5b.GRID, e5b.K, e5b.LK, e5b.SY, e5b.SETA, e5b.T, e5b.TS
AGENTS = {"FULL": dict(seta=SETA, policy="variance", plastic=True), "A1_no_action": dict(seta=SETA, policy="random", plastic=True),
          "A2_no_drift": dict(seta=0.0, policy="variance", plastic=False), "A3_no_pc_feedback": dict(seta=SETA, policy="variance", plastic=False),
          "A4_none": dict(seta=0.0, policy="random", plastic=False), "CONTROL_wrong_sy": dict(seta=SETA, policy="variance", plastic=True, sy_assumed=0.5 * SY)}


def make_env(seed, n):
    rng = np.random.default_rng(seed)
    f = (LK @ rng.standard_normal((G, n))).T
    F = np.empty((T, n, G))
    for t in range(T):
        f = f + SETA * (LK @ rng.standard_normal((G, n))).T
        if t + 1 == TS:
            f = f + (np.cos(GRID) - np.sin(GRID))[None, :]                     # abrupt global shift
        F[t] = f
    return F, rng.standard_normal((T, n)), rng.integers(0, G, (T, n))


def run_agent(cfg, F, eps, ridx, n):
    ag = GridReflexAgent(n, K, SY, **cfg)
    rmse, cov, z2 = np.empty((T, n)), np.empty((T, n)), np.empty((T, n))
    for t in range(T):
        _, z = ag.step(F[t], eps[t], ridx[t])
        d = ag.m - F[t]
        rmse[t], z2[t] = np.sqrt(np.mean(d ** 2, 1)), z ** 2
        cov[t] = np.mean(np.abs(d) <= 1.96 * ag.posterior_sd(), 1)
    return rmse.T, cov.T, z2.T                                                  # (n, T)


def main(smoke=False):
    seeds, n = ([0], 10) if smoke else (list(range(10)), 200)
    res = {a: dict(rmse=[], cov=[], z2=[]) for a in AGENTS}
    for s in seeds:
        F, eps, ridx = make_env(s, n)
        for a, cfg in AGENTS.items():
            r, c, z = run_agent(cfg, F, eps, ridx, n)
            res[a]["rmse"].append(r); res[a]["cov"].append(c); res[a]["z2"].append(z)
    R = {a: {k: np.stack(v) for k, v in d.items()} for a, d in res.items()}      # (seeds, n, T)
    if smoke:
        assert all(np.isfinite(v).all() and v.shape == (1, 10, T) for d in R.values() for v in d.values())
        print("smoke ok: all agents ran, finite arrays of shape (1, 10, 200); no criteria evaluated")
        return
    rows = []
    add = lambda n_, v, c, p, info=False: rows.append((n_ + (" [info]" if info else ""), float(v), c, bool(p), info))  # noqa: E731
    per_seed = lambda a, lo, hi: R[a]["rmse"][:, :, lo:hi].mean((1, 2))          # noqa: E731
    for other, tag in (("A1_no_action", "C1"), ("A2_no_drift", "C2")):
        t_, p_ = stats.ttest_rel(per_seed("FULL", 0, T), per_seed(other, 0, T))
        add(f"{tag} FULL RMSE - {other} (mean over t=1..200)", per_seed("FULL", 0, T).mean() - per_seed(other, 0, T).mean(), f"<0 and p<0.01 (p={p_:.1e})", t_ < 0 and p_ < 0.01)
    rec = {a: e5b.recovery(np.concatenate(list(R[a]["rmse"]), 0)) for a in AGENTS}
    add("C3 median recovery FULL / A3", rec["FULL"][0] / rec["A3_no_pc_feedback"][0] if np.isfinite(rec["A3_no_pc_feedback"][0]) else 0.0,
        f"<=0.8 (FULL {rec['FULL'][0]}, A3 {rec['A3_no_pc_feedback'][0]} steps)", rec["FULL"][0] <= 0.8 * rec["A3_no_pc_feedback"][0])
    d = (per_seed("FULL", 19, 99) - per_seed("A3_no_pc_feedback", 19, 99)) / per_seed("A3_no_pc_feedback", 19, 99)
    ub = d.mean() + stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    add("C4 upper 95% CI of relative RMSE difference FULL vs A3 (t=20..99)", ub, "<+0.05", ub < 0.05)
    cv = R["FULL"]["cov"][:, :, 19:99].mean()
    add("C5 FULL cov95 pre-shift (t=20..99)", cv, "|v-0.95|<0.03", abs(cv - 0.95) < 0.03)
    z2p, z2s = R["A3_no_pc_feedback"]["z2"][:, :, 19:99], R["A3_no_pc_feedback"]["z2"][:, :, 100:105]
    se = np.sqrt(2 / z2p.size)
    add("C6 A3 mean z^2 after shift (t=101..105)", z2s.mean(), ">3", z2s.mean() > 3)
    add("C6 A3 mean z^2 pre-shift (t=20..99)", z2p.mean(), f"|v-1|<{3*se:.4f}", abs(z2p.mean() - 1) < 3 * se)
    cz = R["CONTROL_wrong_sy"]["z2"][:, :, 19:99].mean()
    add("CONTROL wrong sigma_y: mean z^2 (t=20..99)", cz, ">1.5 (must fail calibration)", cz > 1.5)
    base = per_seed("A4_none", 0, T).mean()
    add("INFO FULL vs A4 (none) relative RMSE", per_seed("FULL", 0, T).mean() / base - 1, "info", True, True)
    for a in AGENTS:
        add(f"INFO {a} mean RMSE (t=1..200)", per_seed(a, 0, T).mean(), "info", True, True)
        add(f"INFO {a} median recovery steps", rec[a][0], "info", True, True)

    print(f"{'check':70s}{'value':>10s}  criterion")
    for n_, v, c, p, info in rows:
        print(f"{n_:70s}{v:10.4f}  {c:48s}{('PASS' if p else 'FAIL') + (' (info)' if info else '')}")
    ok = all(r[3] for r in rows if not r[4])
    print(f"\nchecks: {sum(r[3] for r in rows if not r[4])}/{sum(1 for r in rows if not r[4])} pass")
    print("E7 (closed reflex loop, exact-grid reference agent):", "PASS" if ok else "FAIL")

    tt = np.arange(1, T + 1)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for a in ("FULL", "A1_no_action", "A2_no_drift", "A3_no_pc_feedback", "A4_none"):
        ax[0].plot(tt, R[a]["rmse"].mean((0, 1)), label=a)
    ax[0].axvline(TS, color="k", ls=":"); ax[0].set_yscale("log"); ax[0].set_title("grid RMSE"); ax[0].legend(fontsize=7)
    for a in ("FULL", "A3_no_pc_feedback", "CONTROL_wrong_sy"):
        ax[1].plot(tt, R[a]["z2"].mean((0, 1)), label=a)
    ax[1].axvline(TS, color="k", ls=":"); ax[1].set_yscale("log"); ax[1].set_title("mean z^2 (1 = calibrated)"); ax[1].legend(fontsize=7)
    ax[2].bar(list(AGENTS)[:5], [min(rec[a][0], 100) for a in list(AGENTS)[:5]]); ax[2].set_title("median recovery steps (capped at 100)")
    ax[2].tick_params(axis="x", labelrotation=60, labelsize=7)
    fig.tight_layout(); fig.savefig(OUT / "e7_closed_loop.png", dpi=140); plt.close(fig)
    (OUT / "e7_results.json").write_text(json.dumps(dict(
        scope="closed reflex loop, exact-grid reference agent; not the paper's LUT/ellipsoidal mechanisms; engines pending", overall_pass=ok,
        checks=[dict(name=n_, value=v, criterion=c, passed=p, informational=i) for n_, v, c, p, i in rows]), indent=2))


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
