# E4b protocol (DRAFT, awaiting approval): the look-up table principle (paper 2605.02825, sections 5-6)

Scope if it passes: the look-up-table (LUT) mechanism reproduces exact sequential GP simulation in a 1-D, noise-free, known-hyperparameter Gaussian emulator, and the paper's stability and rate claims are (or are not) supported there. It does NOT validate the Student-t version, 2-D forcing inputs, RGPs, or the Bayesian reflex loop. E4 (inducing points) does not transfer: inducing-point methods optimise/marginalise the auxiliary values, the LUT simulates D* and conditions on it.

## What the paper says (claims to test, no numbers are given in the chapter)
- Sec 5: dynamic emulator y_t = eta(v_t), v_t = (z_t, y_{t-1}), eta ~ GP, training D = eta(g_1..g_n). Auxiliary D* = eta(G*) on a fixed grid G* (N points). Given D*, eta(v_t) depends on (D, D*, v_t) only (Markov, "approximately"). Algorithm: (1) y_1 ~ [eta(v_1)|D]; (2) D* ~ [D*|D, y_1]; (3) y_t ~ [eta(v_t)|D, D*], t>=2. The covariance A of (D*, D) is fixed and inverted ONCE; integrating D* out needs a new random matrix inverse every step and, per the paper (attributed to Bhattacharya 2007), "breaks down completely" for long sequences while the LUT stays stable.
- Sec 6.2: approximation of the function by the conditional given D*_n is O(n^-1), n = grid size (Ghosh et al. 2014 theorem).

## Formulation (ours)
eta ~ GP(0, k), k = sf^2 exp(-(v-v')^2/(2 l^2)), sf=1, l=1 (paper: Gaussian correlation). Control kernel: exponential k = sf^2 exp(-|v-v'|/l). Hyperparameters fixed; Gaussian conditionals (paper: Student-t, unknown variance; deferred to E4b-t). Training: n=8 noise-free points. Dynamics y_t = eta(y_{t-1}), y_0=-2, truth map eta*(y)=0.6 sin(y)+0.4 (stable fixed point, so inputs cluster: the stress case for matrix conditioning). Grid G* = N equispaced points on [-4,4] (diagnostic: fraction of steps with v_t outside G* < 1e-3). Jitter 1e-8 on diag(A) for BOTH LUT and the exact oracle.
- EXACT (marginalised) oracle: y_t drawn from eta conditioned on D and all previous (v_s, y_s): exact joint by the chain rule. Three implementations: naive inverse each step (np.linalg.inv, jitter 0, T<=500), Cholesky append jitter 0, Cholesky append jitter 1e-8.
- LUT: A = Cov([D; D*]) + jitter, A^-1 precomputed once (v_1 = y_0 is fixed, so the Step-2 conditional is also precomputed); per step: mean k_v^T A^-1 [D;D*], variance k(v,v) - k_v^T A^-1 k_v, cost O((n+N)^2), independent of t.
- Closed form for fixed inputs v_1..v_T: LUT joint is Gaussian with mean = exact mean and Cov(y_s,y_t) = B_s S B_t^T (s!=t), Var(y_t) = B_t S B_t^T + r_t, Cov(y_1,y_t) = Cov(y_1,D*) B_t^T, where [y_1, D*] ~ exact posterior, B_t = k_{v_t}^T A^-1 restricted to D*, r_t its residual variance. So KL(exact || LUT) is computable without simulation.

## Tests and pre-registered criteria
- T1 precompute integrity (N in {20, 100}): ||A A^-1 - I||_F / sqrt(n+N) < 1e-6; cond(A) reported.
- T2 approximation rate, closed form, N in {5,10,20,40,80,160}, kernels RBF and EXP, T=10 fixed inputs (posterior-mean trajectory): r_N = max_{v in [-3,3]} Var(eta(v)|D,D*) and KL_N.
  - C2a RBF: r_N non-increasing and r_160 < 1e-6.
  - C2b EXP (control, prediction Theta(1/N)): log-log slope of r_N over N=10..160 in [-1.2, -0.8].
  - C2c paper claim "at least O(1/N)": slope of r_N <= -0.9 for BOTH kernels over the range where r_N > 1e-7.
  - C2d KL_N non-increasing in N (down to 1e-12) for both kernels; slopes reported.
  - CONTROL coarse grid (RBF): KL_5 > 1e-2 and KL_5 > 100 x KL_80.
- T3 algorithm validation (N=100, R=20000 replicates, T=10 fixed inputs): sample means and covariances vs the analytic LUT joint, all |diff|/SE < 4.5; y_1 marginal equals the exact one.
- T4 long-stream stability (dynamic, R=20 replicates (10 for naive), T in {20,50,100,500,1000,5000}, N=100). Failure = non-finite value, non-positive pivot, variance < -1e-8, or |y_t|>10.
  - C4a LUT failure rate = 0 at every T.  C4b LUT per-step time at T=5000 / T=100 < 2.
  - C4c LUT vs exact (jitter 1e-8) at T=10, R=5000: two-sample KS p > 0.001 at every t=1..10.
  - INFO (claim check): failure rate, first failing step and cond number of each exact variant; per-step time slope vs t. Report "claim reproduced" only if an exact variant fails by T=1000 while LUT does not; if the jittered Cholesky survives, say the claim is implementation-dependent.
- T5 calibration by simulation (SBC): draw eta ~ GP prior on an 801-point grid, train on 8 points, truth trajectory by interpolation, 400 problems, 200 forecast draws each; 90% central interval coverage at t = 1, 2, 5, 10 within 3 SE (+/-0.043) for EXACT (harness check) and for LUT (N=100).

## Deliverables
src/bayesian_reflex/models/lookup_table.py; experiments/e4b_lookup_table.py; tests/test_lookup_table.py; results/e4b_results.json, e4b_rate_stability.png.

## Decisions needed from you
1. Approve fixed hyperparameters and the 1-D autonomous map first (Student-t and 2-D (z_t, y_{t-1}) later).
2. Approve the EXP-kernel control and the jitter policy (1e-8 for both methods).
3. Keep T5 (SBC) in scope or defer it.
4. Approve the thresholds above (any changes now, not after seeing results).
