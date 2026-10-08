"""E4b v2 diagnostics. Protocol: docs/e4b_v2_diagnostic_protocol.md (declared before running; v1 stays recorded).
Run: python experiments/e4b_v2_diagnostics.py   (~1 min)"""
import json
import pathlib
import sys

import numpy as np
from scipy import stats
from scipy.linalg import cho_factor, cho_solve, solve_triangular

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import e4b_lookup_table as E  # noqa: E402
from bayesian_reflex.models.lookup_table import LookupTableEmulator, exp_kernel, gauss_kernel, kl_gauss  # noqa: E402

OUT = ROOT / "results" / "e4b_v2"
OUT.mkdir(parents=True, exist_ok=True)
rows = []


def add(name, value, crit, ok, info=False):
    rows.append(dict(name=name, value=None if value is None else float(value), criterion=crit, status="PASS" if ok else "FAIL", info=info))


# ---- V1 ----
for N in (20, 100, 200):
    inv, chol, cond = LookupTableEmulator(E.XD, E.YD, E.grid_cc(N), gauss_kernel).integrity()
    add(f"V1a N={N} Cholesky residual", chol, "<1e-6", chol < 1e-6)
    add(f"V1b N={N} explicit-inverse residual vs 100*cond*eps", inv, f"<= {100 * cond * 2.2e-16:.1e} (cond {cond:.1e})", inv <= 100 * cond * 2.2e-16)

# ---- V2 ----
ETA2 = lambda y: 0.5 * np.sin(1.3 * y) + 0.3 * np.cos(0.7 * y) + 0.1  # noqa: E731
mesh = np.linspace(-3, 3, 2001)[:, None]
floors = {}
for tname, f in (("eta*", E.ETA), ("eta2", ETA2)):
    for jit in (1e-8, 1e-10, 1e-12):
        errs = []
        for n in (20, 40, 80, 160, 320):
            G = np.linspace(-3, 3, n + 1)[:-1, None] + 3.0 / n
            try:
                c = cho_factor(gauss_kernel(G, G) + jit * np.eye(n), lower=True)
                errs.append(float(np.abs(gauss_kernel(mesh, G) @ cho_solve(c, f(G[:, 0])) - f(mesh[:, 0])).max()))
            except np.linalg.LinAlgError:
                errs.append(np.nan)
        e = np.array(errs[1:])                                      # n >= 40
        floors[(tname, jit)] = (float(np.nanmin(e)), float(np.nanmax(e) / np.nanmin(e)), errs)
        add(f"V2 {tname} jitter {jit:g}: floor (min over n>=40), plateau ratio {floors[(tname, jit)][1]:.2f}", floors[(tname, jit)][0], "info", True, True)
    add(f"V2a {tname} floor at jitter 1e-8 <= 3e-4", floors[(tname, 1e-8)][0], "<=3e-4", floors[(tname, 1e-8)][0] <= 3e-4)
    r1, r2 = floors[(tname, 1e-8)][0] / floors[(tname, 1e-10)][0], floors[(tname, 1e-10)][0] / floors[(tname, 1e-12)][0]
    add(f"V2b {tname} floor ratios 1e-8/1e-10 and 1e-10/1e-12 (min)", min(r1, r2), ">=3", min(r1, r2) >= 3)
    pl = max(floors[(tname, j)][1] for j in (1e-8, 1e-10, 1e-12))
    add(f"V2c {tname} plateau max/min over n>=40 (worst jitter)", pl, "<=3", pl <= 3)

# ---- V3 ----
ns, z = [20, 40, 80, 160, 320, 640, 1280], np.linspace(-3, 3, 20001)
zc = z[:, None]
res = {k: [] for k in ("int_sup", "int_L2", "bnd_sup", "var")}
for n in ns:
    h = 6.0 / n
    G = np.linspace(-3, 3, n + 1)[:-1, None] + h / 2
    c = cho_factor(exp_kernel(G, G) + 1e-8 * np.eye(n), lower=True)
    Ks = exp_kernel(zc, G)
    err = np.abs(Ks @ cho_solve(c, E.ETA(G[:, 0])) - E.ETA(z))
    inter, bnd = np.abs(z) <= 2.5, np.abs(z) > 3 - 1.5 * h
    res["int_sup"].append(float(err[inter].max())); res["int_L2"].append(float(np.sqrt(np.trapezoid(err[inter] ** 2, z[inter]))))
    res["bnd_sup"].append(float(err[bnd].max()))
    res["var"].append(float((1.0 - (solve_triangular(c[0], Ks.T, lower=True) ** 2).sum(0)).max()))
sl = {k: E.slope(ns, v, floor=1e-9) for k, v in res.items()}
slv = E.slope(ns[2:], res["var"][2:], floor=1e-9)
add("V3a exponential interior sup mean-error slope", sl["int_sup"], "in [-2.3,-1.7]", sl["int_sup"] is not None and -2.3 <= sl["int_sup"] <= -1.7)
add("V3b exponential interior L2 slope", sl["int_L2"], "in [-2.3,-1.7]", sl["int_L2"] is not None and -2.3 <= sl["int_L2"] <= -1.7)
add("V3c exponential boundary-strip sup slope", sl["bnd_sup"], "in [-1.3,-0.7]", sl["bnd_sup"] is not None and -1.3 <= sl["bnd_sup"] <= -0.7)
add("V3d exponential variance slope (n>=80)", slv, "<=-0.95", slv is not None and slv <= -0.95)

# ---- V4 ----
jit = 1e-12
pts = []
for N in (20, 40, 80, 160):
    e = LookupTableEmulator(E.XD, E.YD, E.grid_cc(N), gauss_kernel, jitter=jit)
    for d in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        V = np.array([[-2.0], [-2.0 + d]])
        mx, Cx = e.joint_exact(V); ml, Cl = e.joint_lut(V)
        kl = kl_gauss(mx, Cx, ml, Cl, jit=1e-12)
        s2 = gauss_kernel(V[1:], e.P)
        r = float(e.coeffs(s2)[2][0])
        pts.append((N, d, kl, max(r, 1e-30) / d ** 2))
P = np.array([p for p in pts if p[2] > 1e-12])
rho = stats.spearmanr(np.log(P[:, 2]), np.log(P[:, 3])).statistic
add(f"V4a Spearman(log KL, log r/Delta^2) over {len(P)} pairs", rho, ">0.95", rho > 0.95)
Vs, Ns = np.linspace(-2.4, 2.9, 6)[:, None], [5, 10, 20, 40, 80, 160]
for kn in ("gauss", "exp"):
    kls = []
    for N in Ns:
        e = LookupTableEmulator(E.XD, E.YD, E.grid_cc(N), E.KERN[kn], jitter=1e-12)
        mx, Cx = e.joint_exact(Vs); ml, Cl = e.joint_lut(Vs)
        kls.append(float(kl_gauss(mx, Cx, ml, Cl, jit=1e-12)))
    if kn == "gauss":
        add("V4b Gaussian separated-input KL non-increasing and KL_80<1e-9", kls[4], "non-increasing (tol 1e-11), <1e-9", E.nonincreasing(kls, tol=1e-11) and kls[4] < 1e-9)
        add("V4b CONTROL coarse grid KL_5", kls[0], ">1e-2 and >100*KL_80", kls[0] > 1e-2 and kls[0] > 100 * kls[4])
    else:
        add("V4b exponential separated-input KL, max over N>=20", max(kls[2:]), "<=1e-10", max(kls[2:]) <= 1e-10)
    rows[-1]["kl_curve"] = kls
k12, k10 = E.kl_curve("gauss", [80], jit=1e-12, kljit=1e-12)[0], E.kl_curve("gauss", [80], jit=1e-10, kljit=1e-10)[0]
add("V4c clustered-trajectory KL_80: jitter 1e-12 / 1e-10", k12 / k10, "in [0.5,2]", 0.5 <= k12 / k10 <= 2)

for r in rows:
    print(f"  {r['name'][:84]:84s}{(r['value'] if r['value'] is not None else float('nan')):11.3g}  {r['criterion'][:44]:44s}{r['status']}{' (info)' if r['info'] else ''}")
core = [r for r in rows if not r["info"]]
print(f"\nv2 diagnostic checks: {sum(r['status'] == 'PASS' for r in core)}/{len(core)} pass; misses: {[r['name'] for r in core if r['status'] != 'PASS']}")
(OUT / "e4b_v2_results.json").write_text(json.dumps(dict(rows=rows, v3=dict(ns=ns, **res, slopes=sl), v4a_pairs=pts), indent=2, default=float))
