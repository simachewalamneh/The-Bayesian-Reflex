"""E6 v2 confirmation on FRESH seeds (5000..5199, none used in E6 or in the A1 diagnosis), exact GP, T=200.

Context: original A1 (variance MSE < random at T=100) missed; diagnosis (e6_diagnose_a1.py) showed an early advantage that
vanishes by T~60. v2 criteria are therefore POST-HOC and are tested OUT OF SAMPLE here. Original A1 stays recorded as a miss.
Criteria (fixed before running):
  V2-A1  early query efficiency, tau in {0.02, 0.01}: T_random/T_variance >= 1.2 AND bootstrap 95% CI lower bound > 1
         (T_tau = first t where the seed-mean MSE < tau; seeds resampled 2000x). tau=0.005 is informational (predicted ~1).
  V2-A1r replication at T=30: paired t-test p<0.01 with variance MSE < random MSE.
  V2-A2  non-inferiority: upper 95% CI bound of (MSE_variance - MSE_random)/MSE_random < +10% at T=100 and at T=200.
Run: python experiments/e6_v2_confirmation.py   (~3-8 min)
"""
import json
import pathlib
import sys

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
import e6_uncertainty_action as e  # noqa: E402

SEEDS, B = list(range(5000, 5200)), 2000
rv = np.array([e.run(e.GPEngine, "variance", s)["mse"] for s in SEEDS])
rr = np.array([e.run(e.GPEngine, "random", s)["mse"] for s in SEEDS])
S = len(SEEDS)
rng = np.random.default_rng(0)
idx = rng.integers(0, S, (B, S))
rows = []
add = lambda n, v, c, p, info=False: rows.append((n + (" [info]" if info else ""), v, c, bool(p), info))  # noqa: E731
for thr in (0.02, 0.01, 0.005):
    tv, tr = e.first_below(rv.mean(0), thr), e.first_below(rr.mean(0), thr)
    ratio = tr / tv if np.isfinite(tv) and np.isfinite(tr) else float("nan")
    bs = []
    for b in range(B):
        a, c = e.first_below(rv[idx[b]].mean(0), thr), e.first_below(rr[idx[b]].mean(0), thr)
        bs.append(c / a if np.isfinite(a) and np.isfinite(c) else np.nan)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    print(f"tau={thr}: variance {tv}, random {tr}, ratio {ratio:.2f}, bootstrap 95% CI [{lo:.2f}, {hi:.2f}]")
    if thr == 0.005:
        add(f"V2-A1 ratio tau={thr}", ratio, f"info (CI [{lo:.2f},{hi:.2f}]; predicted ~1)", True, True)
    else:
        add(f"V2-A1 ratio tau={thr}", ratio, f">=1.2 and CI lower>1 (CI [{lo:.2f},{hi:.2f}])", ratio >= 1.2 and lo > 1.0)
d = rv[:, 29] - rr[:, 29]
p = stats.ttest_rel(rv[:, 29], rr[:, 29]).pvalue
add("V2-A1r T=30 mean MSE diff (variance - random)", d.mean(), f"<0 and p<0.01 (p={p:.1e}, rel {d.mean()/rr[:, 29].mean():+.1%})", d.mean() < 0 and p < 0.01)
for tt in (100, 200):
    d = rv[:, tt - 1] - rr[:, tt - 1]
    se, base = d.std(ddof=1) / np.sqrt(S), rr[:, tt - 1].mean()
    lo, hi = (d.mean() - 1.96 * se) / base, (d.mean() + 1.96 * se) / base
    print(f"T={tt}: relative MSE difference {d.mean()/base:+.1%}, 95% CI [{lo:+.1%}, {hi:+.1%}]")
    add(f"V2-A2 non-inferiority T={tt} (upper CI bound)", hi, "<+0.10", hi < 0.10)
print(f"\n{'check':52s}{'value':>10s}  criterion")
for n_, v, c, p_, info in rows:
    print(f"{n_:52s}{v:10.3f}  {c:62s}{('PASS' if p_ else 'FAIL') + (' (info)' if info else '')}")
ok = all(r[3] for r in rows if not r[4])
print(f"\nchecks: {sum(r[3] for r in rows if not r[4])}/{sum(1 for r in rows if not r[4])} pass")
print("E6 v2 (out-of-sample, fresh seeds):", "PASS" if ok else "FAIL")
(ROOT / "results" / "e6_v2_results.json").write_text(json.dumps(dict(
    scope="post-hoc v2 criteria tested on fresh seeds; original A1 (T=100) stays a recorded miss", seeds="5000..5199", overall_pass=ok,
    checks=[dict(name=n_, value=float(v), criterion=c, passed=p_, informational=i) for n_, v, c, p_, i in rows]), indent=2))
