# E4b protocol v2 (DRAFT, awaiting approval): the look-up table principle — paper-to-protocol audit

Sources read: chapter arXiv 2605.02825 sec 5-6 (the look-up table, the marginalised alternative) and Ghosh, Mukhopadhyay, Roy, Bhattacharya, arXiv 1108.3262 (sec 3-4 and Supplement S-1, Theorem S-1.1). No numbers are reported in the chapter; its stability claim is attributed to Bhattacharya (2007), which was not read. All thresholds below are ours.
Scope if it passes: the LUT reproduces exact sequential GP simulation in a Gaussian, known-hyperparameter emulator, and the claims below are supported or not in the settings tested. It does NOT cover Student-t/unknown-variance versions, estimated smoothness (MCMC), the RGP, or the reflex loop; E4 (inducing points) does not transfer.

## Audit: paper statement -> our object -> status
| Paper (where) | Exact content | Our implementation | Status |
|---|---|---|---|
| G* and D* (chapter 5; 1108.3262 4.2) | G* = fixed grid; D* = GP values on it, simulated given eta(v_1) | grid on [-4,4]; D* drawn from the exact conditional (Cholesky) | matches. 4.3 requires v_1 NOT in G* (else D* carries y_1): we assert it |
| "look-up"/"interpolation" (3.4, 4.4) | no nearest-neighbour lookup: eta(v_t) is drawn from the GP conditional given D* (kriging mean/variance, eq. 21-22) | conditional normal with A^-1 | matches; "lookup" is a metaphor. Table values act only through conditioning |
| one-time inverse (4.8) | A_{g,D*}^-1 fixed, computed before simulation/MCMC | precompute A^-1 (jitter 1e-8); Step-2 conditional also precomputable since v_1 is known | matches |
| Markov approximation (eq. 20) | [g(v)\|D*, x_{t-1}..x_1] ~ [g(v)\|D*, x_{t-1}], "arbitrarily accurate" for fine grids | tested via closed-form KL(exact \|\| LUT) (T2b) | THE quantity the algorithm relies on |
| marginalised alternative (4.7-4.8) | integrating D* out makes x_{t+1} depend on all past states; instability from near-singular matrices of clustered states, "particularly if sigma_g^2 and sigma_eta^2 are small"; worse for large t | EXACT oracle = sequential conditioning (naive inverse; Cholesky append, jitter 0 and 1e-8) | matches; claim is a heuristic argument, no experiment in these texts. Mechanism depends on the nugget: add a nugget sweep (T4c) |
| O(1/n) (4.4, Theorem S-1.1) | for an EQUISPACED 1-D grid of n base points, augmented with the true orbits g_true^(k)(z_i), and conditioning on D* only: (1) integrated L^r error of E[g^(t)|D*] vs g_true^(t) is O(1/n); (2) pointwise mean error O(1/n); (3) Var[g^(t)|D*] is O(1/n). Proof: zero at grid points + bounded derivative (mean value theorem) | T2a replicates (1)-(3) for t=1 | see caveats below |
| Student-t, regression mean, estimated R (chapter 5.3; 3.1) | unknown scale/coefficients integrated -> Student-t; correlation parameters by MCMC | known scale, zero mean, fixed length-scale (Gaussian conditionals) | DEFERRED (E4b-t) |

## Caveats found in the audit (what the theorem does and does not say)
1. Theorem S-1.1 bounds the interpolation error of the conditional MEAN and VARIANCE given D*. It does not bound the divergence between the Markov-approximated and the true conditional law (eq. 20), which the paper infers from it. Our T2b measures that divergence directly.
2. The theorem assumes the grid contains the true orbits of the base points, which needs g_true; the practical grid (stratified random points, e.g. n=100 on [-30,30] while the true series sits in [0.8,1.5]) satisfies neither equispacing nor orbit-augmentation. Grid-range mismatch is tested in T2c.
3. The theorem is 1-D; the practical inputs are 2-D (time, state). 2-D behaviour is exploratory (T2e).
4. For smooth truth the mean error with a rough kernel is expected to be better than the bound (piecewise-linear interpolation gives O(n^-2)); with the Gaussian kernel the error should decay much faster than O(1/n) until the jitter floor. "O(1/n)" is therefore expected to hold as an upper bound, not as a sharp rate.
5. In the state-space paper D* is a latent variable updated in MCMC; the forward simulation of the chapter (our test) is the Bhattacharya (2007) emulation setting.

## Formulation (ours)
eta ~ GP(0,k), k = sf^2 exp(-(v-v')^2/(2 l^2)), sf=1, l=1 (paper: exp{-(z1-z2)'R(z1-z2)}); control kernel: exponential exp(-|v-v'|/l). Truth map eta*(y)=0.6 sin(y)+0.4 (stable fixed point, clusters inputs). Training n=8 noise-free points. G* = N equispaced points on [-4,4]. y_0=-2. Jitter 1e-8 on diag(A) for LUT and exact. Fixed-input closed form: LUT joint is Gaussian with the exact mean; Cov(y_s,y_t)=B_s S B_t^T, Var(y_t)=B_t S B_t^T + r_t, Cov(y_1,y_t)=Cov(y_1,D*) B_t^T, where [y_1,D*] ~ exact posterior and (B_t, r_t) are the conditional-mean map and residual variance given (D,D*).

## Tests and thresholds (to be frozen on approval)
- T1 precompute integrity (N=20,100): ||A A^-1 - I||_F/sqrt(n+N) < 1e-6; cond(A) reported.
- T2a theorem replication (t=1, conditioning on D* only, truth eta*, n in {5,10,20,40,80,160,320} equispaced on [-3,3], kernels Gaussian and exponential): quantities (1) L^2 mean error, (2) sup mean error, (3) sup variance. Fit slopes only on values > 1e-6.
  - Gaussian: all three non-increasing and < 1e-6 by n=40 (no slope claim: floor).
  - Exponential (control): sup variance slope in [-1.2,-0.8]; sup and L^2 mean error slopes in [-2.3,-1.7] (predicted n^-2 for smooth truth).
  - Paper claim "at least O(1/n)": every fitted slope <= -0.9 for both kernels.
- T2b Markov approximation: KL_N(exact || LUT), T=10 fixed inputs (posterior-mean trajectory), N in {5,...,160}: RBF: non-increasing and KL_80 < 1e-6; exponential: non-increasing (slope reported); CONTROL coarse grid (RBF): KL_5 > 1e-2 and > 100 x KL_80.
- T2c grid-range mismatch (info, predictions recorded): N=100 on [-30,30] vs [-4,4]; r(v) over the trajectory region.
- T2d (optional, info): t=2 composition variance by Monte Carlo.  T2e (exploratory, info): 2-D (time, state) grid, slope vs total points and vs points per axis; predicted exponential-kernel variance slopes -0.5 and -1.
- T3 algorithm validation (N=100, R=20000, fixed inputs, T=10): sample means/covariances vs analytic LUT joint, all |diff|/SE < 4.5; assert v_1 not in G* (min distance > 1e-6).
- T4 long-stream stability (R=20, naive R=10, T in {20,50,100,500,1000,5000}, N=100); failure = non-finite, non-positive pivot, variance < -1e-8, or |y_t|>10. Settings: (a) autonomous noise-free map; (b) time-indexed 2-D inputs v_t=(t, y_{t-1}); (c) state-space nugget sweep sigma_eta in {0, 1e-3, 1e-1} on (a).
  - C4a LUT failure rate = 0 in every setting and T.  C4b LUT per-step time T=5000 / T=100 < 2.
  - C4c LUT vs exact (jitter 1e-8) at T=10, R=5000, setting (a): KS p > 0.001 at every t=1..10.
  - PREDICTIONS (reported, not pass/fail): exact variants fail without jitter in (a); fewer failures in (b); failure rate non-increasing in the nugget in (c). "Claim reproduced" only if an exact variant fails by T=1000 while the LUT does not.
- T5 SBC (keep or defer; your call): eta ~ GP prior on an 801-point grid, 8 training points, 400 problems x 200 forecast draws, 90% interval coverage at t=1,2,5,10 within +/-0.043 for EXACT and LUT (N=100).

## Decisions needed
1. Approve the audit corrections (theorem scope, orbit-augmented grid, 1-D) and the added T2b/T2c/T4b/T4c.
2. Approve the predictions-versus-thresholds split in T2a and T4 (slopes for the exponential control; Gaussian only floor-limited).
3. T5 in or out.  4. Freeze the thresholds now (no changes after seeing results).

## Implementation amendments (declared BEFORE the full run; thresholds above are unchanged)
- A1 T2b: the frozen quantity (KL for the posterior-mean trajectory inputs) is the primary pass/fail. The same KL for 10 well-separated inputs is reported as info (the trajectory inputs cluster, so C_ex is near-singular).
- A2 "non-increasing" checks allow the floor tolerance v_{k+1} <= v_k (1+1e-3) + 1e-8.
- A3 One-time precomputation = the Cholesky factor of A (equivalent to A^-1; used for simulation, C4a, C4b). The explicit inverse is formed for T1 and run as a variant `lut_inv` (info). Disclosure: in the smoke run (tiny sizes) the explicit-inverse variant failed within a few steps at N=100, jitter 1e-8, so the explicit inverse is not usable as written; this was seen before the full run.
- A4 Grid nodes are cell-centred with a fixed offset of 0.0731 of a cell, so that no node coincides with v_1 for any N (N=10 cell centres would contain y_0=-2).
- A5 Slope fits use >= 2 points above 1e-6.
- A6 Noise sweep: sigma_eta is a standard deviation (nugget variance 1e-6 and 1e-2); the sigma_eta=0 case is setting (a).
- A7 Compute: LUT runs to T=5000 in (a),(c); exact variants run to T=1000 (naive to 500), extendable with --tmax-exact; setting (b) runs every method to T=500 with N=500 grid points (100 time nodes x 5 state nodes), length-scales (10, 1), 24 training points, truth eta*(t,y)=0.6 sin y + 0.4 + 0.3 cos(0.2 t).
- A8 Disclosure from development: with jitter 1e-8 the Gaussian-kernel KL plateaus (KL_80 = 2.6e-5 in a trial run) because C_ex has eigenvalues down to 1e-7, comparable to the jitter; at jitter 1e-10 / 1e-12 it was 4e-9 / 6e-13. The frozen criterion (KL_80 < 1e-6 at jitter 1e-8) is therefore expected to FAIL; the jitter sweep is added as an info diagnostic. KL uses jitter 1e-9 on both covariances in the primary calculation.
