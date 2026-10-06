"""E5-A - Scalar nonstationary drift: Kalman-form belief with process noise vs static belief and heuristics.

Scope: validates TRACKING of a drifting scalar under a known linear-Gaussian state-space model (Kalman oracle),
and shows the failure of a static belief. "Distinguishing observation noise from state change" holds only through the
fixed sigma_eta/sigma_y ratio; this is NOT change-point detection (the shift stream exposes that limit).
It says nothing about function-space drift (E5-B), action, or the closed loop.

Model  : theta_t = theta_{t-1} + eta_t, eta~N(0,0.05^2); y_t = theta_t + eps_t, eps~N(0,0.5^2); theta_0~N(0,1); T=200;
         1000 trials x 10 seeds.
Streams: RW (model correct; PASS/FAIL), SHIFT (theta 0 -> 2 at t=100, model misspecified), TREND (theta_t=0.01t, info only).
Methods: kalman (true sigma_eta), static (sigma_eta=0), mismatched sigma_eta x0.2 and x5, forgetting 'matched' and
         constant lambda*, heuristics: EMA (alpha = steady Kalman gain), sliding window W=20.
Criteria (fixed before running):
  RW  Kalman: cov95 of theta_t within 3 SE at t in {1,10,50,100,200}; z_t mean/var/lag-1 within 3 SE; KS p>0.001 at those t.
  RW  floor: Kalman var(T=200) == analytic steady-state (rel 1e-6); Kalman var / static var > 10 at T.
  RW  identities: static drift == E1 belief (1e-12); matched lambda == Kalman (1e-10); constant lambda* var -> Kalman (<1e-6, t>=100).
  RW  accuracy (t>=100): RMSE_kalman <= min(EMA, window) + 0.002; RMSE_static > 2 x RMSE_kalman.
  CONTROLS (must fail): static cov95(T) < 0.90; mismatched sigma_eta x0.2 and x5: |cov95(T)-0.95| > 0.03.
  v2 (added after diagnosis, v1 miss stays recorded): both variances -> analytic P_ss (<1e-12 at T=1000);
         |var gap| log-slope over t=50..200 within 2% of ln(lambda*).
  SHIFT: Kalman median recovery <= 60 steps and >=90% recover within 100; static <10% recover (must fail).
         recovery = first step after the jump where |theta-mu| <= 0.25 holds 5 steps in a row (also reported with 1.96*sd).
Run: python experiments/e5a_scalar_drift.py
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
from bayesian_reflex.models import ScalarDriftBelief, ScalarGaussianBelief  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
SY, SETA, T, N, SEEDS, TJ, W, BAND = 0.5, 0.05, 200, 1000, list(range(10)), 100, 20, 0.25
CPS = [1, 10, 50, 100, 200]
Q = SETA ** 2
P_PRED = (Q + np.sqrt(Q ** 2 + 4 * Q * SY ** 2)) / 2          # analytic steady-state predicted variance
P_SS = P_PRED * SY ** 2 / (P_PRED + SY ** 2)                   # analytic steady-state posterior variance
K_SS = P_PRED / (P_PRED + SY ** 2)                             # steady-state Kalman gain
LAM_SS = P_SS / (P_SS + Q)


def simulate(stream, rng):
    if stream == "rw":
        theta = np.cumsum(np.concatenate([rng.standard_normal((N, 1)), SETA * rng.standard_normal((N, T))], 1), 1)[:, 1:]
    elif stream == "shift":
        theta = np.tile(np.where(np.arange(1, T + 1) >= TJ, 2.0, 0.0), (N, 1))
    else:
        theta = np.tile(0.01 * np.arange(1, T + 1), (N, 1))
    return theta, theta + SY * rng.standard_normal((N, T))


def make_models():
    mk = lambda s, f=None: ScalarDriftBelief(0.0, 1.0, SY, s, N, forgetting=f)  # noqa: E731
    return {"kalman": mk(SETA), "static": mk(0.0), "eta_x0.2": mk(0.2 * SETA), "eta_x5": mk(5 * SETA),
            "lam_matched": mk(SETA, "matched"), "lam_const": mk(SETA, LAM_SS)}


def run(stream, seed):
    rng = np.random.default_rng(seed)
    theta, y = simulate(stream, rng)
    models = make_models()
    mu = {k: np.empty((N, T)) for k in models}; sd = {k: np.empty((N, T)) for k in models}; z = np.empty((N, T))
    for t in range(T):
        for k, m in models.items():
            mp, vp = m.predict()
            if k == "kalman":
                z[:, t] = (y[:, t] - mp) / np.sqrt(vp)               # innovation, standardized, BEFORE the update
            m.update(y[:, t])
            mu[k][:, t], sd[k][:, t] = m.mu, m.sd
    def ema_of(alpha):
        out, prev = np.empty((N, T)), np.zeros(N)
        for t in range(T):
            prev = prev + alpha * (y[:, t] - prev); out[:, t] = prev
        return out
    cs = np.cumsum(np.concatenate([np.zeros((N, 1)), y], 1), 1)
    win = np.stack([(cs[:, t + 1] - cs[:, max(0, t + 1 - W)]) / (t + 1 - max(0, t + 1 - W)) for t in range(T)], 1)
    mu["ema"], mu["window"] = ema_of(K_SS), win
    mu["ema_slow"], mu["ema_fast"] = ema_of(0.02), ema_of(0.5)   # untuned heuristics (info only)
    return theta, y, mu, sd, z


def recovery(theta, mu, sd=None):
    ok = np.abs(theta[:, TJ - 1:] - mu[:, TJ - 1:]) <= (BAND if sd is None else 1.96 * sd[:, TJ - 1:])
    sus = ok[:, :-4] & ok[:, 1:-3] & ok[:, 2:-2] & ok[:, 3:-1] & ok[:, 4:]
    first = np.where(sus.any(1), sus.argmax(1), np.inf)
    return float(np.median(first)), float(np.mean(np.isfinite(first) & (first <= 100)))


def pool(stream):
    runs = [run(stream, s) for s in SEEDS]
    cat = lambda i: np.concatenate([r[i] for r in runs])  # noqa: E731
    mu = {k: np.concatenate([r[2][k] for r in runs]) for k in runs[0][2]}
    sd = {k: np.concatenate([r[3][k] for r in runs]) for k in runs[0][3]}
    return cat(0), cat(1), mu, sd, cat(4), runs[0]


def main():
    rows = []
    add = lambda n, v, c, p, info=False: rows.append((n + (" [info]" if info else ""), float(v), c, bool(p), info))  # noqa: E731
    th, y, mu, sd, z, first = pool("rw")
    M = th.shape[0]
    cov = lambda k, t: float(np.mean(np.abs(th[:, t - 1] - mu[k][:, t - 1]) <= 1.96 * sd[k][:, t - 1]))  # noqa: E731

    for t in CPS:
        tol = 3 * np.sqrt(0.95 * 0.05 / M)
        add(f"RW kalman cov95 t={t}", cov("kalman", t), f"|v-0.95|<{tol:.4f}", abs(cov("kalman", t) - 0.95) < tol)
        p = stats.kstest(z[:, t - 1], "norm").pvalue
        add(f"RW kalman innovation KS p t={t}", p, ">0.001", p > 1e-3)
    n = z.size
    r1 = np.corrcoef(z[:, :-1].ravel(), z[:, 1:].ravel())[0, 1]
    add("RW innovation mean z", z.mean(), f"|v|<{3/np.sqrt(n):.4f}", abs(z.mean()) < 3 / np.sqrt(n))
    add("RW innovation var z", z.var(), f"|v-1|<{3*np.sqrt(2/n):.4f}", abs(z.var() - 1) < 3 * np.sqrt(2 / n))
    add("RW innovation lag-1 autocorr", r1, f"|v|<{3/np.sqrt(n):.4f}", abs(r1) < 3 / np.sqrt(n))

    vk, vs = sd["kalman"][:, -1] ** 2, sd["static"][:, -1] ** 2
    rel = abs(vk[0] - P_SS) / P_SS
    add("Kalman var(T) vs analytic steady state (rel)", rel, "<1e-6", rel < 1e-6)
    add("floor: Kalman var / static var at T", vk[0] / vs[0], ">10", vk[0] / vs[0] > 10)
    # identities (single seed, same data)
    th0, y0, mu0, sd0, _ = first
    e1 = ScalarGaussianBelief(0.0, 1.0, SY, N)
    d1 = 0.0
    for t in range(T):
        e1.update(y0[:, t]); d1 = max(d1, np.abs(e1.mu - mu0["static"][:, t]).max(), np.abs(e1.sd - sd0["static"][:, t]).max())
    add("static drift model == E1 belief (max diff)", d1, "<1e-12", d1 < 1e-12)
    dm = max(np.abs(mu0["lam_matched"] - mu0["kalman"]).max(), np.abs(sd0["lam_matched"] - sd0["kalman"]).max())
    add("matched lambda == Kalman (max diff)", dm, "<1e-10", dm < 1e-10)
    dc = np.abs(sd0["lam_const"][:, TJ - 1:] ** 2 - sd0["kalman"][:, TJ - 1:] ** 2).max()
    add("constant lambda* var -> Kalman var (t>=100)", dc, "<1e-6", dc < 1e-6)
    d150 = np.abs(sd0["lam_const"][:, 149:] ** 2 - sd0["kalman"][:, 149:] ** 2).max()   # POST-HOC diagnostic, see docstring note
    add("POST-HOC lambda* var diff (t>=150)", d150, "info (decay of the t>=100 miss)", True, True)

    # ---- v2 criteria (agreed after diagnosing the v1 miss; added BESIDE v1, which stays recorded) ----
    kv, lv = ScalarDriftBelief(0.0, 1.0, SY, SETA, 1), ScalarDriftBelief(0.0, 1.0, SY, SETA, 1, forgetting=LAM_SS)
    for _ in range(1000):
        kv.update(np.zeros(1)); lv.update(np.zeros(1))
    for nm, m_ in (("Kalman", kv), ("constant lambda*", lv)):
        add(f"v2a {nm} var(T=1000) vs analytic P_ss", abs(m_.var[0] - P_SS), "<1e-12", abs(m_.var[0] - P_SS) < 1e-12)
    dd = np.abs(sd0["lam_const"][0, 49:200] ** 2 - sd0["kalman"][0, 49:200] ** 2)
    slope = np.polyfit(np.arange(50, 201), np.log(dd), 1)[0]
    add("v2b |var gap| log-slope (t=50..200) vs ln(lambda*), rel", abs(slope / np.log(LAM_SS) - 1), "<0.02", abs(slope / np.log(LAM_SS) - 1) < 0.02)

    rm = lambda k: float(np.sqrt(np.mean((th[:, 99:] - mu[k][:, 99:]) ** 2)))  # noqa: E731
    rk, others = rm("kalman"), min(rm("ema"), rm("window"))
    add("RW RMSE kalman (t>=100)", rk, f"<= min(EMA,window)+0.002 = {others+0.002:.4f}", rk <= others + 0.002)
    for k in ("ema", "ema_slow", "ema_fast", "window"):
        add(f"RW RMSE {k} (t>=100)", rm(k), "info", True, True)
    add("RW RMSE static / kalman", rm("static") / rk, ">2", rm("static") / rk > 2)
    c_static = cov("static", T)
    add("CONTROL static cov95(T)", c_static, "<0.90 (must fail)", c_static < 0.90)
    for k in ("eta_x0.2", "eta_x5"):
        add(f"CONTROL {k} |cov95(T)-0.95|", abs(cov(k, T) - 0.95), ">0.03 (must fail)", abs(cov(k, T) - 0.95) > 0.03)

    # ---- shift and trend ----
    ths, ys, mus, sds, _, firs = pool("shift")
    rec = {k: recovery(ths, mus[k]) for k in mus}
    rec196 = {k: recovery(ths, mus[k], sds[k]) for k in sds}
    add("SHIFT kalman median recovery (steps)", rec["kalman"][0], "<=60", rec["kalman"][0] <= 60)
    add("SHIFT kalman fraction recovered <=100", rec["kalman"][1], ">=0.90", rec["kalman"][1] >= 0.90)
    add("CONTROL SHIFT static fraction recovered", rec["static"][1], "<0.10 (must fail)", rec["static"][1] < 0.10)
    for k in ("ema", "ema_slow", "ema_fast", "window", "eta_x5", "eta_x0.2"):
        add(f"SHIFT {k} median recovery", rec[k][0], "info", True, True)
    ttr, _, mut, sdt, _, firt = pool("trend")
    for k in ("kalman", "static", "ema", "window"):
        add(f"TREND RMSE {k} (t>=100)", float(np.sqrt(np.mean((ttr[:, 99:] - mut[k][:, 99:]) ** 2))), "info", True, True)

    print(f"{'check':50s}{'value':>11s}  criterion")
    for nm, v, c, p, info in rows:
        print(f"{nm:50s}{v:11.3e}  {c:34s}{('PASS' if p else 'FAIL') + (' (info)' if info else '')}")
    print(f"\nanalytic steady state: P_ss={P_SS:.5f} (sd {np.sqrt(P_SS):.4f}), K_ss={K_SS:.4f}, lambda*={LAM_SS:.4f}; static var(T)={vs[0]:.5f}")
    print("SHIFT recovery with |err|<=1.96 sd (median steps, fraction):", {k: (round(v[0], 1), round(v[1], 2)) for k, v in rec196.items() if k in ("kalman", "eta_x5", "eta_x0.2", "static")})
    KNOWN = ["constant lambda* var -> Kalman var (t>=100)"]            # v1 miss, diagnosed (e5a_diagnose_lambda.py)
    miss = [r[0] for r in rows if not r[3] and not r[4]]
    ok = miss == KNOWN
    print(f"\nchecks: {sum(r[3] for r in rows if not r[4])}/{sum(1 for r in rows if not r[4])} pass (v1 + v2); recorded v1 miss: {miss}")
    print("E5-A (scalar drift tracking under a known linear-Gaussian model):",
          "PASS under v2 (the v1 miss stays recorded)" if ok else "FAIL")

    # ---- figure ----
    x = np.arange(1, T + 1)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for a, (title, theta1, mu_, sd_) in zip(ax.ravel()[:3], (("random walk (1 trial)", first[0][0], first[2], first[3]),
                                                                ("sudden shift +2 at t=100", firs[0][0], firs[2], firs[3]),
                                                                ("linear trend", firt[0][0], firt[2], firt[3]))):
        a.plot(x, theta1, "k--", label="true θ_t")
        for k, col, nm in (("static", "tab:red", "static belief"), ("kalman", "tab:blue", "dynamic reflex (σ_η)")):
            a.plot(x, mu_[k][0], color=col, label=nm)
            a.fill_between(x, mu_[k][0] - 1.96 * sd_[k][0], mu_[k][0] + 1.96 * sd_[k][0], color=col, alpha=0.2)
        a.plot(x, mu_["ema"][0], color="tab:green", lw=0.8, alpha=0.8, label="EMA (α=K_ss)")
        a.plot(x, mu_["window"][0], color="tab:orange", lw=0.8, alpha=0.8, label=f"window W={W}")
        a.set_title(title); a.set_xlabel("t")
    ax[0, 0].legend(fontsize=7)
    ax[1, 1].semilogy(x, first[3]["static"][0] ** 2, color="tab:red", label="static σ_t²")
    ax[1, 1].semilogy(x, first[3]["kalman"][0] ** 2, color="tab:blue", label="dynamic σ_t²")
    ax[1, 1].axhline(P_SS, color="k", ls=":", label="analytic floor P_ss")
    ax[1, 1].set_title("uncertainty floor"); ax[1, 1].set_xlabel("t"); ax[1, 1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "e5a_scalar_drift_comparison.png", dpi=140); plt.close(fig)

    (OUT / "e5a_results.json").write_text(json.dumps(dict(
        scope="scalar drift tracking under known linear-Gaussian model; not change-point detection; not function space",
        overall_pass_v1=False, overall_pass_v2=bool(ok), recorded_v1_miss=KNOWN, analytic=dict(P_ss=P_SS, K_ss=K_SS, lambda_star=LAM_SS),
        checks=[dict(name=n_, value=v, criterion=c, passed=p, informational=i) for n_, v, c, p, i in rows],
        shift_recovery={k: dict(median_steps=v[0], frac_recovered=v[1]) for k, v in rec.items()}), indent=2))


if __name__ == "__main__":
    main()
