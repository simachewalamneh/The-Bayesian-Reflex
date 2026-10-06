"""E5-A diagnostic: why does constant-lambda* variance differ from Kalman variance by 1.058e-6 at t=100?
Deterministic variance recursions only (variance does not depend on the data), longdouble, start P0=s0^2=1.
Tests five hypotheses: (1) finite-horizon convergence-rate difference, (2) P_t vs P_{t|t-1}, (3) definition of lambda,
(4) numerical accumulation, (5) tolerance. Run: python experiments/e5a_diagnose_lambda.py
"""
import numpy as np

LD = np.longdouble
SY, SETA, S0, T = LD(0.5), LD(0.05), LD(1.0), 1000
s2, q = SY ** 2, SETA ** 2
Pp = (q + np.sqrt(q ** 2 + 4 * q * s2)) / 2                 # steady predicted var P_{t|t-1}
Pss = Pp * s2 / (Pp + s2)                                    # steady posterior var P_t
K = Pp / (Pp + s2)
lam = Pss / (Pss + q)
print(f"(3) definition: lambda*=P/(P+q)={float(lam):.10f}  1-K_ss={float(1-K):.10f}  identity |diff|={float(abs(lam-(1-K))):.1e}")

fK = lambda P: (P + q) * s2 / (P + q + s2)                   # Kalman posterior-variance map
fL = lambda P: (P / lam) * s2 / (P / lam + s2)               # constant-lambda map
PK, PL = np.empty(T, LD), np.empty(T, LD)
a = b = S0 ** 2
for t in range(T):
    a, b = fK(a), fL(b); PK[t], PL[t] = a, b
d = np.abs(PL - PK)
print(f"(1) fixed points: Kalman {float(PK[-1]):.12f}  lambda* {float(PL[-1]):.12f}  analytic {float(Pss):.12f}")
print(f"    |diff| at t=100: {float(d[99]):.4e}  (experiment reported 1.058e-06);  max over t>=100: {float(d[99:].max()):.4e}")
t_ok = int(np.argmax(d < 1e-6)) + 1
print(f"    first t with |diff|<1e-6: t={t_ok};  diff at t=150: {float(d[149]):.2e}, t=200: {float(d[199]):.2e}")
dK = (1 - K) ** 2                                            # d fK/dP at fixed point
dL = lam * s2 ** 2 / (Pss + lam * s2) ** 2                   # d fL/dP at fixed point
print(f"    local contraction rates: Kalman (1-K)^2={float(dK):.4f}, constant-lambda={float(dL):.4f} (= lambda* ? {float(abs(dL-lam)):.1e})")
sl = np.polyfit(np.arange(50, 200), np.log(np.maximum(d[50:200].astype(float), 1e-300)), 1)[0]
print(f"    empirical log-slope of |diff| over t=50..200: {sl:.4f}  vs ln(lambda*)={float(np.log(lam)):.4f}")
print(f"(2) variances compared are posterior P_t for both; predicted P_pred(lambda)=P/lambda={float(Pss/lam):.6f} vs Kalman P+q={float(Pss+q):.6f}")
d64 = np.abs(np.array([0.0]))                                # (4) float64 vs longdouble
a = b = 1.0
f64K = lambda P: (P + float(q)) * float(s2) / (P + float(q) + float(s2))
f64L = lambda P: (P / float(lam)) * float(s2) / (P / float(lam) + float(s2))
x = y = 1.0; mx = 0.0
for t in range(100):
    x, y = f64K(x), f64L(y)
    mx = max(mx, abs(x - float(PK[t])), abs(y - float(PL[t])))
print(f"(4) float64 vs longdouble recursion max error (t<=100): {mx:.1e}  (<< 1e-6, so not numerical)")
