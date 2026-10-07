"""E7 engine confirmation: exact-grid (reduced N), PC-FSVI stand-in (reduced and full N), real FSVI+pc_infer (reduced N).
Protocol and low-power handling: docs/e7_confirmation_protocol.md (declared before running; thresholds identical to E7).
Run: python experiments/e7_engines.py            (~6-8 min, all engines)
     or one engine at a time: --only exact_reduced | standin_reduced | standin_full | real_reduced, then --summary
     python experiments/e7_engines.py --smoke    (1 seed x 3 trials per engine, executes only)
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
import e7_closed_loop as e7  # noqa: E402
from bayesian_reflex.models.grid_reflex import GridReflexAgent, InducingGridAgent, RealGridAgent  # noqa: E402

OUT = ROOT / "results"
T, TS, SY, K, GRID = e7.T, e7.TS, e7.SY, e7.K, e7.GRID


def build(engine, cfg, n):
    cfg = dict(cfg)
    seta = cfg.pop("seta")
    sy_a = cfg.pop("sy_assumed", None)
    common = dict(sy_true=SY, seta=seta, sy_assumed=sy_a, **cfg)
    if engine == "exact":
        return GridReflexAgent(n, K, **common)
    return (InducingGridAgent if engine == "standin" else RealGridAgent)(n, grid=GRID, **common)


def run(engine, seeds, n):
    R = {a: dict(rmse=[], cov=[], z2=[]) for a in e7.AGENTS}
    for s in seeds:
        F, eps, ridx = e7.make_env(s, n)
        for a, cfg in e7.AGENTS.items():
            ag = build(engine, cfg, n)
            rmse, cov, z2 = np.empty((T, n)), np.empty((T, n)), np.empty((T, n))
            for t in range(T):
                _, z = ag.step(F[t], eps[t], ridx[t])
                mu, sd = (ag.m, ag.posterior_sd()) if engine == "exact" else ag.mean_sd()
                d = mu - F[t]
                rmse[t], z2[t], cov[t] = np.sqrt(np.mean(d ** 2, 1)), z ** 2, np.mean(np.abs(d) <= 1.96 * sd, 1)
            R[a]["rmse"].append(rmse.T); R[a]["cov"].append(cov.T); R[a]["z2"].append(z2.T)
    return {a: {k: np.stack(v) for k, v in d.items()} for a, d in R.items()}


def evaluate(R):
    rows = []                                                      # (name, value, criterion, status)
    ps = lambda a, lo, hi: R[a]["rmse"][:, :, lo:hi].mean((1, 2))  # noqa: E731
    for other, tag in (("A1_no_action", "C1"), ("A2_no_drift", "C2")):
        t_, p_ = stats.ttest_rel(ps("FULL", 0, T), ps(other, 0, T))
        st = "PASS" if (t_ < 0 and p_ < 0.01) else "INCONCLUSIVE" if (t_ < 0 and p_ < 0.10) else "FAIL"
        rows.append((f"{tag} FULL - {other} RMSE", ps("FULL", 0, T).mean() - ps(other, 0, T).mean(), f"<0, p<0.01 (p={p_:.1e})", st))
    rec = {a: e5b.recovery(np.concatenate(list(R[a]["rmse"]), 0)) for a in e7.AGENTS}
    fa, a3 = rec["FULL"][0], rec["A3_no_pc_feedback"][0]
    rows.append(("C3 median recovery FULL / A3", fa / a3 if np.isfinite(a3) else 0.0, f"<=0.8 (FULL {fa}, A3 {a3})", "PASS" if fa <= 0.8 * a3 else "FAIL"))
    d = (ps("FULL", 19, 99) - ps("A3_no_pc_feedback", 19, 99)) / ps("A3_no_pc_feedback", 19, 99)
    ub = d.mean() + stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    rows.append(("C4 upper CI rel. RMSE FULL vs A3 (t=20..99)", ub, "<0.05", "PASS" if ub < 0.05 else "INCONCLUSIVE" if ub < 0.10 else "FAIL"))
    cv = R["FULL"]["cov"][:, :, 19:99].mean()
    rows.append(("C5 FULL cov95 pre-shift", cv, "|v-0.95|<0.03", "PASS" if abs(cv - 0.95) < 0.03 else "FAIL"))
    zp, zs = R["A3_no_pc_feedback"]["z2"][:, :, 19:99], R["A3_no_pc_feedback"]["z2"][:, :, 100:105]
    se = np.sqrt(2 / zp.size)
    rows.append(("C6 A3 mean z^2 after shift", zs.mean(), ">3", "PASS" if zs.mean() > 3 else "FAIL"))
    rows.append(("C6 A3 mean z^2 pre-shift", zp.mean(), f"|v-1|<{3*se:.4f}", "PASS" if abs(zp.mean() - 1) < 3 * se else "FAIL"))
    cz = R["CONTROL_wrong_sy"]["z2"][:, :, 19:99].mean()
    rows.append(("CONTROL wrong sigma_y mean z^2", cz, ">1.5 (must fail calibration)", "PASS" if cz > 1.5 else "FAIL"))
    sts = [r[3] for r in rows]
    verdict = "FAIL" if "FAIL" in sts else "INCONCLUSIVE" if "INCONCLUSIVE" in sts else "PASS"
    return rows, verdict, {a: float(ps(a, 0, T).mean()) for a in e7.AGENTS}, rec


def run_item(name, plan):
    eng, _, seeds, n = [x for x in plan if x[1] == name][0]
    R = run(eng, seeds, n)
    rows, verdict, rm, rec = evaluate(R)
    res = dict(verdict=verdict, seeds=len(seeds), trials=n, mean_rmse=rm, recovery={a: float(v[0]) for a, v in rec.items()},
               checks=[dict(name=r[0], value=float(r[1]), criterion=r[2], status=r[3]) for r in rows],
               curve_full_rmse=R["FULL"]["rmse"].mean((0, 1)).tolist(), curve_a3_z2=R["A3_no_pc_feedback"]["z2"].mean((0, 1)).tolist())
    (OUT / f"e7_engines_{name}.json").write_text(json.dumps(res, indent=2))
    print(f"\n== {name} ({len(seeds)} seeds x {n} trials): {verdict}")
    for r in rows:
        print(f"  {r[0]:46s}{r[1]:10.4f}  {r[2]:42s}{r[3]}")
    print("  mean RMSE:", {a: round(v, 3) for a, v in rm.items()}, "| median recovery:", {a: res["recovery"][a] for a in ("FULL", "A3_no_pc_feedback")})


def summary(plan):
    out = {p[1]: json.loads((OUT / f"e7_engines_{p[1]}.json").read_text()) for p in plan}
    print("verdicts:", {k: v["verdict"] for k, v in out.items()})
    ok = out["standin_reduced"]["verdict"] == "PASS" and out["real_reduced"]["verdict"] == "PASS"
    print("E7 conclusion survives engine substitution (stand-in and real both PASS):", "YES" if ok else "NO / NOT ESTABLISHED")
    tt = np.arange(1, T + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, v in out.items():
        ax[0].plot(tt, v["curve_full_rmse"], label=name); ax[1].plot(tt, v["curve_a3_z2"], label=name)
    ax[0].axvline(TS, color="k", ls=":"); ax[0].set_yscale("log"); ax[0].set_title("FULL agent grid RMSE"); ax[0].legend(fontsize=7)
    ax[1].axvline(TS, color="k", ls=":"); ax[1].set_yscale("log"); ax[1].set_title("A3 mean z^2"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "e7_engines.png", dpi=140); plt.close(fig)
    (OUT / "e7_engines_results.json").write_text(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.startswith("curve")} for k, v in out.items()}, indent=2))


def main(smoke=False):
    plan = [("exact", "exact_reduced", list(range(5)), 40), ("standin", "standin_reduced", list(range(5)), 40),
            ("standin", "standin_full", list(range(10)), 200), ("real", "real_reduced", list(range(5)), 40)]
    if smoke:
        for eng, name, _, _ in plan[:2] + plan[3:]:
            R = run(eng, [0], 3)
            assert all(np.isfinite(v).all() for d in R.values() for v in d.values())
        print("smoke ok: exact, stand-in and real engines executed; no criteria evaluated")
    elif "--only" in sys.argv:
        run_item(sys.argv[sys.argv.index("--only") + 1], plan)
    elif "--summary" in sys.argv:
        summary(plan)
    else:
        for p in plan:
            run_item(p[1], plan)
        summary(plan)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv)
