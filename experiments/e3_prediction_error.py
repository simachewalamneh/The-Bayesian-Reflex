"""E3 - One-step-ahead prediction error, standardized by the Bayesian predictive (scalar Gaussian model).

Scope: validates that one-step-ahead prediction errors are correctly standardized by the Bayesian predictive
uncertainty under the scalar Gaussian model. It does NOT establish that a predictive-coding layer itself
performs or drives Bayesian learning (that interface decision is still open).

z_t = (y_t - yhat_t) / sqrt(sigma_{t-1}^2 + sigma_y^2), yhat_t and sigma_{t-1}^2 from the belief BEFORE seeing y_t.
Setup  : theta ~ prior N(0,1); 1000 trials x 10 seeds x T=50; sigma_y in {0.1, 1.0}.
Checks (correct model): mean z ~ 0, var z ~ 1 (3 SE), KS vs N(0,1) at t=1,5,25,50 (p>0.001), lag-1 autocorr ~ 0 (3 SE),
        Kalman identity  mu_t = mu_{t-1} + K_t e_t, K_t = s2_{t-1}/(s2_{t-1}+sy^2)  to <1e-12,
        precision weighting: raw error scale follows sigma_y (ratio > 5) while z scale does not (ratio 1 +/- 0.01).
Negative controls (must FAIL): wrong sigma_y (0.5x), stale predictive variance (posterior after update).
Extension (paper 5.2): unknown noise variance (Normal-Inverse-Gamma) -> z ~ Student-t(2a_{t-1}); controls: N(0,1) CDF must fail.
Run: python experiments/e3_prediction_error.py
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
from bayesian_reflex import PredictiveCodingLayer  # noqa: E402
from bayesian_reflex.models import ScalarGaussianBelief  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
C = dict(n=1000, seeds=list(range(10)), T=50, sys=[0.1, 1.0], cps=[1, 5, 25, 50],
         wrong_factor=0.5, a0=1.5, b0=1.0, k0=1.0, m0=0.0)


def gaussian_run(sy, seed):
    rng = np.random.default_rng(seed)
    n, T = C["n"], C["T"]
    theta = rng.standard_normal(n)                                   # theta ~ prior N(0,1)
    y = theta[:, None] + sy * rng.standard_normal((n, T))
    b = ScalarGaussianBelief(0.0, 1.0, sy, n)
    bw = ScalarGaussianBelief(0.0, 1.0, C["wrong_factor"] * sy, n)   # control: wrong noise level
    pc = PredictiveCodingLayer()
    Z, E, Zs, Zw = (np.empty((n, T)) for _ in range(4))
    K, kal = np.empty(T), 0.0
    for t in range(T):
        yt = y[:, t]
        mu_p, v_pred = b.predict()                                   # BEFORE the update
        var_prev = b.var
        E[:, t] = pc.prediction_error(yt, mu_p)
        Z[:, t] = pc.standardized_error(yt, mu_p, v_pred)
        mw, vw = bw.predict()
        Zw[:, t] = pc.standardized_error(yt, mw, vw)
        gain = var_prev / (var_prev + sy ** 2)
        b.update(yt)
        kal = max(kal, float(np.abs(b.mu - (mu_p + gain * E[:, t])).max()))
        Zs[:, t] = pc.standardized_error(yt, mu_p, b.var + sy ** 2)  # control: stale (post-update) variance
        bw.update(yt)
        K[t] = gain[0]
    return Z, E, Zs, Zw, kal, K


def nig_run(seed):
    """Unknown variance: y~N(theta,s2), theta|s2~N(m,s2/k), s2~IG(a,b). Predictive = Student-t(2a, m, b(k+1)/(a k))."""
    rng = np.random.default_rng(1000 + seed)
    n, T, a, b, k, m = C["n"], C["T"], C["a0"], np.full(C["n"], C["b0"]), C["k0"], np.full(C["n"], C["m0"])
    s2 = C["b0"] / rng.gamma(C["a0"], 1.0, n)
    theta = C["m0"] + np.sqrt(s2 / C["k0"]) * rng.standard_normal(n)
    y = theta[:, None] + np.sqrt(s2)[:, None] * rng.standard_normal((n, T))
    Z, DF = np.empty((n, T)), np.empty(T)
    for t in range(T):
        scale = np.sqrt(b * (k + 1) / (a * k))
        Z[:, t], DF[t] = (y[:, t] - m) / scale, 2 * a
        k_new = k + 1
        b = b + k * (y[:, t] - m) ** 2 / (2 * k_new)
        m = (k * m + y[:, t]) / k_new
        k, a = k_new, a + 0.5
    return Z, DF


def main():
    rows, store = [], {}
    for sy in C["sys"]:
        runs = [gaussian_run(sy, s) for s in C["seeds"]]
        Z, E, Zs, Zw = (np.concatenate([r[i] for r in runs]) for i in range(4))
        kal, K = max(r[4] for r in runs), runs[0][5]
        store[sy] = dict(Z=Z, E=E, Zs=Zs, Zw=Zw, K=K)
        M = Z.size
        r1 = np.corrcoef(Z[:, :-1].ravel(), Z[:, 1:].ravel())[0, 1]
        tag = f"sy={sy}"
        rows += [(f"{tag} mean z", Z.mean(), f"|v|<{3/np.sqrt(M):.4f}", abs(Z.mean()) < 3 / np.sqrt(M)),
                 (f"{tag} var z", Z.var(), f"|v-1|<{3*np.sqrt(2/M):.4f}", abs(Z.var() - 1) < 3 * np.sqrt(2 / M)),
                 (f"{tag} lag-1 autocorr", r1, f"|v|<{3/np.sqrt(M):.4f}", abs(r1) < 3 / np.sqrt(M))]
        for t in C["cps"]:
            p = stats.kstest(Z[:, t - 1], "norm").pvalue
            rows.append((f"{tag} KS p t={t}", p, ">0.001", p > 1e-3))
        rows.append((f"{tag} Kalman identity max err", kal, "<1e-12", kal < 1e-12))
        vw = Zw.var()
        rows.append((f"CONTROL {tag} wrong sy: var z", vw, "|v-1|>0.1 (must fail)", abs(vw - 1) > 0.1))
        vs = Zs[:, :5].var()
        rows.append((f"CONTROL {tag} stale var: var z t<=5", vs, "|v-1|>0.05 (must fail)", abs(vs - 1) > 0.05))
        ps = stats.kstest(Zs[:, 0], "norm").pvalue
        rows.append((f"CONTROL {tag} stale var: KS p t=1", ps, "<0.001 (must fail)", ps < 1e-3))

    lo, hi = store[0.1], store[1.0]
    raw_ratio = hi["E"][:, -1].std() / lo["E"][:, -1].std()
    z_ratio = hi["Z"].std() / lo["Z"].std()
    rows += [("precision wt: raw err std ratio (1.0/0.1)", raw_ratio, ">5 (raw follows sy)", raw_ratio > 5),
             ("precision wt: z std ratio (1.0/0.1)", z_ratio, "|v-1|<0.01", abs(z_ratio - 1) < 0.01)]

    # ---- extension: unknown variance, Student-t ----
    nr = [nig_run(s) for s in C["seeds"]]
    Zn, DF = np.concatenate([r[0] for r in nr]), nr[0][1]
    for t in C["cps"]:
        p = stats.kstest(stats.t.cdf(Zn[:, t - 1], DF[t - 1]), "uniform").pvalue
        rows.append((f"NIG Student-t PIT KS p t={t}", p, ">0.001", p > 1e-3))
    pc_ = stats.kstest(stats.norm.cdf(Zn[:, 0]), "uniform").pvalue
    rows.append(("CONTROL NIG normal CDF KS p t=1", pc_, "<0.001 (must fail)", pc_ < 1e-3))

    print(f"{'check':46s}{'value':>11s}  criterion")
    for n_, v, c, p in rows:
        print(f"{n_:46s}{v:11.3e}  {c:26s}{'PASS' if p else 'FAIL'}")
    ok = all(r[3] for r in rows)
    print("\nE3 (scalar Gaussian one-step-ahead standardized prediction error):", "PASS" if ok else "FAIL")

    # ---- figure ----
    q = np.linspace(0.001, 0.999, 200)
    th = stats.norm.ppf(q)
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for a, sy in zip(ax[0, :2], C["sys"]):
        s = store[sy]
        a.plot(th, np.quantile(s["Z"], q), "b-", label="correct")
        a.plot(th, np.quantile(s["Zw"], q), "r--", label="wrong sy (0.5x)")
        a.plot(th, np.quantile(s["Zs"][:, :5], q), color="orange", ls="-.", label="stale var (t<=5)")
        a.plot(th, th, "k:"); a.set_title(f"QQ of z vs N(0,1), sy={sy}")
        a.set_xlabel("N(0,1) quantile"); a.legend(fontsize=7)
    t_ax = np.arange(1, C["T"] + 1)
    for sy, col in zip(C["sys"], ["tab:green", "tab:purple"]):
        ax[0, 2].plot(t_ax, store[sy]["E"].std(0), color=col, ls="--", label=f"raw error std, sy={sy}")
        ax[0, 2].plot(t_ax, store[sy]["Z"].std(0), color=col, label=f"z std, sy={sy}")
        ax[1, 0].plot(t_ax, store[sy]["K"], color=col, label=f"gain K_t, sy={sy}")
    ax[0, 2].set_title("precision weighting: raw error vs z"); ax[0, 2].set_yscale("log"); ax[0, 2].legend(fontsize=7)
    ax[1, 0].set_title("Kalman gain K_t = s2_{t-1}/(s2_{t-1}+sy^2)"); ax[1, 0].set_xlabel("t"); ax[1, 0].legend(fontsize=7)
    ax[1, 1].hist(stats.t.cdf(Zn[:, 0], DF[0]), bins=20, density=True, alpha=0.8)
    ax[1, 1].axhline(1, color="k", ls=":"); ax[1, 1].set_title("NIG t=1 PIT, Student-t CDF (correct)")
    ax[1, 2].hist(stats.norm.cdf(Zn[:, 0]), bins=20, density=True, color="tab:red", alpha=0.8)
    ax[1, 2].axhline(1, color="k", ls=":"); ax[1, 2].set_title("NIG t=1 PIT, N(0,1) CDF (control, must fail)")
    fig.suptitle("E3: standardized one-step-ahead prediction error (scalar Gaussian model)")
    fig.tight_layout(); fig.savefig(OUT / "e3_prediction_error.png", dpi=140); plt.close(fig)

    (OUT / "e3_results.json").write_text(json.dumps(dict(
        scope="scalar Gaussian one-step-ahead prediction errors standardized by the Bayesian predictive; "
              "does not establish that a PC layer performs or drives learning",
        overall_pass=bool(ok),
        checks=[dict(name=n_, value=float(v), criterion=c, passed=bool(p)) for n_, v, c, p in rows]), indent=2))


if __name__ == "__main__":
    main()
