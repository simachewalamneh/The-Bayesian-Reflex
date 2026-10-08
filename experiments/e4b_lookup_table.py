"""E4b - the look-up table principle (paper 2605.02825 sec 5-6; Ghosh et al. 1108.3262 Thm S-1.1). Protocol: docs/e4b_protocol.md v2
(approved, T5 deferred) + the implementation amendments A1-A8 listed there (declared before the full run).
Parts: t1 t2a t2b t2c t2e t3 t4_a t4_b t4_c t4ks.   Run: python experiments/e4b_lookup_table.py --only <part>   then   --summary
       python experiments/e4b_lookup_table.py --smoke   (tiny sizes, executes only)
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
from scipy.linalg import cho_factor, cho_solve, solve_triangular

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bayesian_reflex.models.lookup_table import (LookupTableEmulator, exact_chol_trajectory, exact_naive_trajectory,  # noqa: E402
                                                exp_kernel, gauss_kernel, kl_gauss)

SMOKE = "--smoke" in sys.argv
OUT = pathlib.Path("/tmp/e4b_smoke") if SMOKE else ROOT / "results"      # smoke runs never touch results/
OUT.mkdir(exist_ok=True, parents=True)
ETA = lambda y: 0.6 * np.sin(y) + 0.4                                    # noqa: E731  truth map (autonomous setting)
XD = np.linspace(-3, 3, 8)[:, None]
YD = ETA(XD[:, 0])
KERN = {"gauss": gauss_kernel, "exp": exp_kernel}
OFF = 0.0731                                                              # A4: grid offset so that no node equals v_1


def grid_cc(N, lo=-4.0, hi=4.0):
    return (lo + (np.arange(N) + 0.5 + OFF) * (hi - lo) / N)[:, None]


def slope(ns, vals, floor=1e-6):
    ns, vals = np.asarray(ns, float), np.asarray(vals, float)
    k = vals > floor
    return float(np.polyfit(np.log(ns[k]), np.log(vals[k]), 1)[0]) if k.sum() >= 2 else None


def nonincreasing(vals, tol=1e-8):                                        # A2: floor tolerance
    v = np.asarray(vals, float)
    return bool(np.all(v[1:] <= v[:-1] * (1 + 1e-3) + tol))


def row(name, value, crit, ok, info=False):
    return dict(name=name, value=float(value) if value is not None else None, criterion=crit, status="PASS" if ok else "FAIL", info=info)


def save(part, rows, info):
    (OUT / f"e4b_{part}.json").write_text(json.dumps(dict(rows=rows, info=info), indent=2, default=float))
    for r in rows:
        print(f"  {r['name']:62s}{(r['value'] if r['value'] is not None else float('nan')):12.4g}  {r['criterion']:44s}{r['status']}{' (info)' if r['info'] else ''}")


def part_t1():
    rows = []
    for N in (20, 100):
        e = LookupTableEmulator(XD, YD, grid_cc(N), gauss_kernel)
        inv_res, chol_res, cond = e.integrity()
        rows += [row(f"T1 N={N} ||A A^-1 - I||_F/sqrt(n+N), explicit inverse", inv_res, "<1e-6", inv_res < 1e-6),
                 row(f"T1 N={N} same residual, Cholesky solve (used in simulation)", chol_res, "info", True, True),
                 row(f"T1 N={N} cond(A)", cond, "info", True, True)]
    save("t1", rows, {})


def interp_stats(kern, n, mesh):
    G = np.linspace(-3, 3, n + 1)[:-1, None] + 3.0 / n
    A = kern(G, G) + 1e-8 * np.eye(n)
    c = cho_factor(A, lower=True)
    Ks = kern(mesh, G)
    err = np.abs(Ks @ cho_solve(c, ETA(G[:, 0])) - ETA(mesh[:, 0]))
    var = 1.0 - (solve_triangular(c[0], Ks.T, lower=True) ** 2).sum(0)
    return float(np.sqrt(np.trapezoid(err ** 2, mesh[:, 0]))), float(err.max()), float(var.max())


def part_t2a():
    ns, mesh = [5, 10, 20, 40, 80, 160, 320], np.linspace(-3, 3, 2001)[:, None]
    res = {k: np.array([interp_stats(KERN[k], n, mesh) for n in ns]) for k in KERN}      # (n, [L2, sup mean, sup var])
    rows, names = [], ["L2 mean error", "sup mean error", "sup variance"]
    for j, nm in enumerate(names):
        v = res["gauss"][:, j]
        rows.append(row(f"T2a Gaussian {nm}: non-increasing and <1e-6 at n=40", v[3], "non-increasing, <1e-6", nonincreasing(v) and v[3] < 1e-6))
    sl = {k: [slope(ns, res[k][:, j]) for j in range(3)] for k in KERN}
    rows.append(row("T2a exponential sup-variance slope", sl["exp"][2], "in [-1.2,-0.8]", sl["exp"][2] is not None and -1.2 <= sl["exp"][2] <= -0.8))
    rows.append(row("T2a exponential sup-mean-error slope", sl["exp"][1], "in [-2.3,-1.7]", sl["exp"][1] is not None and -2.3 <= sl["exp"][1] <= -1.7))
    rows.append(row("T2a exponential L2-mean-error slope", sl["exp"][0], "in [-2.3,-1.7]", sl["exp"][0] is not None and -2.3 <= sl["exp"][0] <= -1.7))
    for k in KERN:
        mx = max(s for s in sl[k] if s is not None)
        rows.append(row(f"T2a paper claim 'at least O(1/n)': max fitted slope, {k}", mx, "<=-0.9", mx <= -0.9))
    save("t2a", rows, dict(ns=ns, values={k: res[k].tolist() for k in KERN}, slopes=sl))


def traj_inputs(emu, T=10, y0=-2.0):
    V, y = [], y0
    for _ in range(T):
        V.append(y); y = float(emu.posterior_given_D(np.array([[y]]))[0][0])
    return np.array(V)[:, None]


def kl_curve(kname, Ns, jit=1e-8, kljit=1e-9, separated=False):
    out = []
    for N in Ns:
        e = LookupTableEmulator(XD, YD, grid_cc(N), KERN[kname], jitter=jit)
        V = np.linspace(-2.63, 2.71, 10)[:, None] if separated else traj_inputs(e)
        mx, Cx = e.joint_exact(V); ml, Cl = e.joint_lut(V)
        out.append(float(kl_gauss(mx, Cx, ml, Cl, jit=kljit)))
    return out


def part_t2b():
    Ns = [5, 10, 20, 40, 80, 160]
    kl = {k: kl_curve(k, Ns) for k in KERN}
    rows = [row("T2b Gaussian KL(exact||LUT): non-increasing and KL_80<1e-6", kl["gauss"][4], "non-increasing, <1e-6", nonincreasing(kl["gauss"]) and kl["gauss"][4] < 1e-6),
            row("T2b exponential KL: non-increasing", kl["exp"][4], "non-increasing", nonincreasing(kl["exp"])),
            row("T2b CONTROL coarse grid: KL_5 (Gaussian)", kl["gauss"][0], ">1e-2 and >100*KL_80", kl["gauss"][0] > 1e-2 and kl["gauss"][0] > 100 * kl["gauss"][4]),
            row("T2b exponential KL slope (info)", slope(Ns, kl["exp"], 1e-12), "info", True, True)]
    info = dict(Ns=Ns, kl=kl)
    info["amendment_A1_separated_inputs"] = {k: kl_curve(k, Ns, separated=True) for k in KERN}
    info["jitter_sweep_gauss"] = {f"jit={j:g}": kl_curve("gauss", Ns, jit=j, kljit=min(1e-9, j)) for j in (1e-8, 1e-10, 1e-12)}
    rows.append(row("T2b [diag] Gaussian KL_80 at jitter 1e-8 / 1e-10 / 1e-12", info["jitter_sweep_gauss"]["jit=1e-08"][4], f"info: {[f'{v[4]:.1e}' for v in info['jitter_sweep_gauss'].values()]}", True, True))
    save("t2b", rows, info)


def part_t2c():
    rows, info = [], {}
    for lo, hi in ((-30, 30), (-4, 4)):
        G = grid_cc(100, lo, hi)
        c = cho_factor(gauss_kernel(G, G) + 1e-8 * np.eye(100), lower=True)
        z = np.linspace(-2.0, 1.1, 400)[:, None]
        v = 1.0 - (solve_triangular(c[0], gauss_kernel(z, G).T, lower=True) ** 2).sum(0)
        info[f"[{lo},{hi}]"] = float(v.max())
        rows.append(row(f"T2c N=100 on [{lo},{hi}]: sup variance over [-2,1.1]", v.max(), "info (prediction: both <1e-6)", True, True))
    save("t2c", rows, info)


def part_t2e():
    rng = np.random.default_rng(0)
    test = rng.uniform(-2.5, 2.5, (1500, 2))
    ms, rows, info = [3, 5, 8, 12, 18, 25], [], {}
    for k in KERN:
        v = []
        for m in ms:
            g = (-3 + (np.arange(m) + 0.5) * 6.0 / m)
            G = np.array([(a, b) for a in g for b in g])
            c = cho_factor(KERN[k](G, G) + 1e-8 * np.eye(m * m), lower=True)
            v.append(float((1.0 - (solve_triangular(c[0], KERN[k](test, G).T, lower=True) ** 2).sum(0)).max()))
        info[k] = v
        rows.append(row(f"T2e 2-D {k}: sup-variance slope vs total points N=m^2", slope([m * m for m in ms], v), "info (exp prediction -0.5)", True, True))
        rows.append(row(f"T2e 2-D {k}: sup-variance slope vs points per axis m", slope(ms, v), "info (exp prediction -1)", True, True))
    save("t2e", rows, dict(ms=ms, values=info))


def part_t3():
    R = 2000 if SMOKE else 20000
    e = LookupTableEmulator(XD, YD, grid_cc(100), gauss_kernel)
    V = np.linspace(-2.63, 2.71, 10)[:, None]
    m, C = e.joint_lut(V)
    x, ff = e.simulate(R, 10, np.random.default_rng(0), lambda t, yl: np.tile(V[t - 1], (len(yl), 1)))
    dm = np.abs(x.mean(0) - m) / np.sqrt(np.diag(C) / R)
    S = np.cov(x.T)
    se = np.sqrt((np.outer(np.diag(C), np.diag(C)) + C ** 2) / R)
    dc = (np.abs(S - C) / se)[np.triu_indices(10)]
    rows = [row("T3 v_1 not in grid (min distance > 1e-6)", float(np.min(np.abs(V[0] - e.G))), ">1e-6", np.min(np.abs(V[0] - e.G)) > 1e-6),
            row("T3 max |sample mean - analytic| / SE", dm.max(), "<4.5", dm.max() < 4.5),
            row("T3 max |sample cov - analytic| / SE", dc.max(), "<4.5", dc.max() < 4.5),
            row("T3 simulation failures", int(np.isfinite(ff).sum()), "==0", not np.isfinite(ff).any())]
    save("t3", rows, dict(R=R))


# ---------------- T4 ----------------
INP_A = lambda t, yl: np.asarray(yl, float)[:, None]                                          # noqa: E731
INP_B = lambda t, yl: np.stack([np.full(len(yl), float(t)), np.asarray(yl, float)], 1)        # noqa: E731
ETA_B = lambda X: 0.6 * np.sin(X[:, 1]) + 0.4 + 0.3 * np.cos(0.2 * X[:, 0])                  # noqa: E731


def setting(name):
    if name == "b":
        Xd = np.array([(t, y) for t in (0, 100, 200, 300, 400, 500) for y in (-3, -1, 1, 3)], float)
        kern = lambda A, B: gauss_kernel(A, B, ell=[10.0, 1.0])                                  # noqa: E731
        G = np.array([(2.5 + 5 * i, -4 + (j + 0.5 + OFF) * 1.6) for i in range(100) for j in range(5)])
        return Xd, ETA_B(Xd), kern, G, INP_B, 500
    return XD, YD, gauss_kernel, grid_cc(100), INP_A, 5000


def rate(ff, Tlist):
    return {int(T): float(np.mean(ff <= T)) for T in Tlist}


def run_t4(name, nugget_sd=0.0):
    Xd, yd, kern, G, inp, Tmax = setting(name)
    R = 4 if SMOKE else 20
    Tl = [T for T in (20, 50, 100, 500, 1000, 5000) if T <= Tmax]
    Tex = min(Tmax, int(sys.argv[sys.argv.index("--tmax-exact") + 1]) if "--tmax-exact" in sys.argv else 1000)
    if SMOKE:
        Tmax, Tl, Tex = 100, [20, 50, 100], 100
    nug = nugget_sd ** 2
    res, info = {}, dict(setting=name, nugget_sd=nugget_sd, T_list=Tl, T_lut=Tmax, T_exact=Tex, R=R)
    for solver in ("chol", "inv"):
        e = LookupTableEmulator(Xd, yd, G, kern, nugget=nug, solver=solver)
        x, ff, times = e.simulate(R, Tmax, np.random.default_rng(1), inp, y0=-2.0, time_steps=True)
        res[f"lut_{solver}"] = ff
        if solver == "chol":
            info["step_time_ms"] = {k: float(np.median(times[a - 2:b - 1]) * 1e3) for k, (a, b) in {"t91_100": (91, 100), "t4991_5000": (4991, 5000)}.items() if b - 2 < len(times)}
    for label, jit in (("chol_jit0", 0.0), ("chol_jit1e-8", 1e-8)):
        ff = np.array([exact_chol_trajectory(Xd, yd, kern, -2.0, Tex, inp, np.random.default_rng(100 + r), jit, nug)[1] for r in range(R)])
        res[label] = ff
    Rn, Tn = (2 if SMOKE else 10), min(Tex, 500)
    res["naive_jit0"] = np.array([exact_naive_trajectory(Xd, yd, kern, -2.0, Tn, inp, np.random.default_rng(200 + r), nug)[1] for r in range(Rn)])
    info["fail_rates"] = {k: rate(v, [T for T in Tl if T <= (Tn if k == "naive_jit0" else Tex if k.startswith("chol") else Tmax)]) for k, v in res.items()}
    info["first_fail_median"] = {k: (float(np.median(v[np.isfinite(v)])) if np.isfinite(v).any() else None) for k, v in res.items()}
    return res, info


def part_t4(name, nugget_sd=0.0):
    res, info = run_t4(name, nugget_sd)
    tag = f"T4{name}" + (f" nugget_sd={nugget_sd:g}" if nugget_sd else "")
    lut_bad = int(np.isfinite(res["lut_chol"]).sum())
    rows = [row(f"{tag} C4a LUT (Cholesky precompute) failures over all T", lut_bad, "==0", lut_bad == 0)]
    if name == "a" and not nugget_sd and "t4991_5000" in info["step_time_ms"]:
        ratio = info["step_time_ms"]["t4991_5000"] / info["step_time_ms"]["t91_100"]
        rows.append(row(f"{tag} C4b LUT per-step time ratio T=5000 / T=100", ratio, "<2", ratio < 2))
    for k in ("lut_inv", "chol_jit0", "chol_jit1e-8", "naive_jit0"):
        fr = info["fail_rates"][k]
        rows.append(row(f"{tag} [info] failure rate {k}", max(fr.values()) if fr else 0.0, f"info: {fr}; median first failure {info['first_fail_median'][k]}", True, True))
    save(f"t4_{name}" + (f"{nugget_sd:g}" if nugget_sd else ""), rows, info)


def part_t4ks():
    R = 500 if SMOKE else 5000
    e = LookupTableEmulator(XD, YD, grid_cc(100), gauss_kernel)
    xl, _ = e.simulate(R, 10, np.random.default_rng(7), INP_A, y0=-2.0)
    xe = np.array([exact_chol_trajectory(XD, YD, gauss_kernel, -2.0, 10, INP_A, np.random.default_rng(500 + r), 1e-8)[0] for r in range(R)])
    ps = [float(stats.ks_2samp(xl[:, t], xe[:, t]).pvalue) for t in range(10)]
    save("t4ks", [row("T4 C4c min KS p-value over t=1..10 (LUT vs exact jitter 1e-8)", min(ps), ">0.001", min(ps) > 1e-3)], dict(p_values=ps, R=R))


PARTS = {"t1": part_t1, "t2a": part_t2a, "t2b": part_t2b, "t2c": part_t2c, "t2e": part_t2e, "t3": part_t3, "t4_a": lambda: part_t4("a"),
         "t4_b": lambda: part_t4("b"), "t4_c1": lambda: part_t4("a", 1e-3), "t4_c2": lambda: part_t4("a", 1e-1), "t4ks": part_t4ks}
FILES = {"t1": "t1", "t2a": "t2a", "t2b": "t2b", "t2c": "t2c", "t2e": "t2e", "t3": "t3", "t4_a": "t4_a", "t4_b": "t4_b", "t4_c1": "t4_a0.001", "t4_c2": "t4_a0.1", "t4ks": "t4ks"}


def summary():
    allrows, miss = [], []
    for p, f in FILES.items():
        fp = OUT / f"e4b_{f}.json"
        if not fp.exists():
            miss.append(p); continue
        allrows += json.loads(fp.read_text())["rows"]
    core = [r for r in allrows if not r["info"]]
    for r in allrows:
        print(f"  {r['name']:64s}{(r['value'] if r['value'] is not None else float('nan')):12.4g}  {r['criterion'][:46]:46s}{r['status']}{' (info)' if r['info'] else ''}")
    npass = sum(r["status"] == "PASS" for r in core)
    print(f"\nchecks: {npass}/{len(core)} pass; parts not yet run: {miss}")
    print("misses:", [r["name"] for r in core if r["status"] != "PASS"])
    print("E4b (look-up table principle, Gaussian known-hyperparameter emulator):", "PASS" if npass == len(core) and not miss else "NOT PASSING / INCOMPLETE")
    try:
        t2a, t2b = json.loads((OUT / "e4b_t2a.json").read_text())["info"], json.loads((OUT / "e4b_t2b.json").read_text())["info"]
        ta = json.loads((OUT / "e4b_t4_a.json").read_text())["info"]
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        ns = t2a["ns"]
        for k, ls in (("gauss", "-"), ("exp", "--")):
            for j, nm in enumerate(["L2 mean err", "sup mean err", "sup var"]):
                ax[0].loglog(ns, np.maximum(np.array(t2a["values"][k])[:, j], 1e-12), ls, label=f"{k} {nm}")
        ax[0].loglog(ns, 1.0 / np.array(ns), "k:", label="1/n"); ax[0].legend(fontsize=6); ax[0].set_title("T2a interpolation error vs grid size n")
        for k in KERN:
            ax[1].loglog(t2b["Ns"], np.maximum(t2b["kl"][k], 1e-14), label=f"{k} (jitter 1e-8)")
        for j, v in t2b["jitter_sweep_gauss"].items():
            ax[1].loglog(t2b["Ns"], np.maximum(v, 1e-14), ":", label=f"gauss {j}")
        ax[1].legend(fontsize=6); ax[1].set_title("T2b KL(exact || LUT) vs N")
        for k, v in ta["fail_rates"].items():
            ax[2].plot(list(map(int, v)), list(v.values()), "o-", label=k)
        ax[2].set_xscale("log"); ax[2].set_title("T4a failure rate vs T"); ax[2].legend(fontsize=6)
        fig.tight_layout(); fig.savefig(OUT / "e4b_rate_stability.png", dpi=140); plt.close(fig)
    except FileNotFoundError:
        pass
    (OUT / "e4b_results.json").write_text(json.dumps(dict(checks=allrows, parts_missing=miss), indent=2, default=float))


if __name__ == "__main__":
    if "--summary" in sys.argv:
        summary()
    elif "--only" in sys.argv:
        p = sys.argv[sys.argv.index("--only") + 1]
        print(f"== {p}"); t0 = time.time(); PARTS[p](); print(f"   ({time.time() - t0:.0f}s)")
    elif SMOKE:
        for p, f in PARTS.items():
            PARTS[p]()
        print("smoke ok: all parts executed; no criteria evaluated")
    else:
        print("use --only <part> (t1 t2a t2b t2c t2e t3 t4_a t4_b t4_c1 t4_c2 t4ks), --summary, or --smoke")
