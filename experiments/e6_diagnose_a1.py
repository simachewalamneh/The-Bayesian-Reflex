"""E6 diagnostic for the A1 miss (variance sampling vs random at T=100): 200 paired seeds, exact GP, T=100.
Answers: is A1 an underpowered 20-seed comparison or a real null? Post-hoc; the original A1 criterion stays recorded.
Run: python experiments/e6_diagnose_a1.py   (~2 min)"""
import json
import pathlib
import sys

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import e6_uncertainty_action as e  # noqa: E402

e.T = 100
S = range(200)
rv = np.array([e.run(e.GPEngine, "variance", s)["mse"] for s in S])
rr = np.array([e.run(e.GPEngine, "random", s)["mse"] for s in S])
out = {}
for tt in (10, 20, 30, 45, 60, 80, 100):
    d = rv[:, tt - 1] - rr[:, tt - 1]
    se = d.std(ddof=1) / np.sqrt(len(d))
    out[tt] = dict(variance=float(rv[:, tt - 1].mean()), random=float(rr[:, tt - 1].mean()), rel_diff=float(d.mean() / rr[:, tt - 1].mean()),
                   ci95_rel=[float((d.mean() - 1.96 * se) / rr[:, tt - 1].mean()), float((d.mean() + 1.96 * se) / rr[:, tt - 1].mean())],
                   p_paired_t=float(stats.ttest_rel(rv[:, tt - 1], rr[:, tt - 1]).pvalue), p_wilcoxon=float(stats.wilcoxon(d).pvalue),
                   frac_variance_better=float(np.mean(d < 0)))
    o = out[tt]
    print(f"T={tt:3d}: variance {o['variance']:.5f} random {o['random']:.5f} rel diff {o['rel_diff']:+.1%} (95% CI {o['ci95_rel'][0]:+.1%}..{o['ci95_rel'][1]:+.1%}) "
          f"p_t={o['p_paired_t']:.2g} p_W={o['p_wilcoxon']:.2g} better in {o['frac_variance_better']:.0%} of seeds")
cv, cr = rv.mean(0), rr.mean(0)
thr = {}
for t_ in (0.02, 0.01, 0.005):
    a, b = e.first_below(cv, t_), e.first_below(cr, t_)
    thr[str(t_)] = dict(variance=a, random=b, ratio=(b / a if np.isfinite(a) else None))
    print(f"queries to mean MSE<{t_}: variance {a}, random {b}, ratio {b / a if np.isfinite(a) else float('nan'):.2f}")
dd = rv[:, 99] - rr[:, 99]
need = int(np.ceil(((2.58 + 0.84) * dd.std(ddof=1) / max(abs(dd.mean()), 1e-12)) ** 2))
print(f"paired sd at T=100 {dd.std(ddof=1):.5f}; observed |diff| {abs(dd.mean()):.5f}; seeds for 80% power at alpha=0.01: ~{need}")
(ROOT / "results" / "e6_a1_diagnosis.json").write_text(json.dumps(dict(seeds=200, T=100, by_T=out, thresholds=thr, seeds_needed_80pct_power=need), indent=2))
