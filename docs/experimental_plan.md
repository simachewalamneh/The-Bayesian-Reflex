# Experimental plan (locked order)

Source: arXiv 2608.00492 (idea: Bayesian reflex as predictive coding) and arXiv 2605.02825 (original chapter: look-up table + ellipsoidal decomposition). Principle: every stage has an oracle or analytic reference.

| Stage | Question | Reference / oracle | Status |
|---|---|---|---|
| E1 | Scalar belief, accuracy + SBC calibration | analytic Gaussian posterior | foundation validated (scalar Gaussian only; 32/32 checks) |
| E2a | Sequential = batch, scalar Gaussian (Thm 2.1), T=100 and stress T=1e5 | batch posterior in longdouble | foundation validated (scalar Gaussian only) |
| E2b | Sequential = batch for GP (rank-one update), with E4 | exact batch GP | validated (max diff 2.6e-13, inside E4) |
| E3 | Precision-weighted prediction error z_t ~ N(0,1) | exact predictive | standardization validated (scalar Gaussian only; does not show the PC layer drives learning) |
| E4 | Function space: exact GP vs PC-FSVI | exact GP | stand-in validated; real FSVI+pc_infer also validated in this setting (E4-real) |
| E4b | Look-up table principle (paper sec 5-6) | exact GP | todo |
| E5-A | Scalar drift: tracking, uncertainty floor, shift recovery | Kalman filter | **frozen: PASS under v2** (28/29 checks; the v1 miss is recorded and diagnosed as a convergence-rate difference; v2 criteria added beside it); tracking validated under a known model only |
| E5-B | Functional drift: B1 oracle-correct random walk, B2 global switch, B3 regional change | exact grid Kalman (functional state-space GP) | stand-in: 20/21 checks (B3 retention-ratio criterion missed: 1.54 vs <1.5); real PC-FSVI swap pending |
| E6 | Uncertainty-driven action: A variance, B info gain, C Thompson | random baseline / known regret | todo |
| E7 | Full reflex loop + ablations | composition of validated parts | todo |
| E8a | Ellipsoidal sampler standalone | exact samplers | todo |
| E8b | Ellipsoidal sampler inside the loop (Alg 4) | exact conjugate | todo |
| E8c | Recursive GP / online BNN (optional) | simpler inference | todo |

## Protocol rules
- **A stage validates only the component it tests.** "The Bayesian Reflex" is evaluated only at E7 (closed loop, with ablations).
- Fix pass/fail criteria before coding; >=10 seeds; report mean +/- sd; every positive test has a negative control that must fail.
- E1-E3 are validation tests of the implementation; E4-E8 carry the research claims.

## Decisions recorded
- E1: coverage tolerance = 3 binomial SE; ranks from posterior samples (reusable for non-Gaussian methods).
- E3: z uses the one-step-ahead predictive, var = sigma_{t-1}^2 + sigma_y^2, computed before the update.
- E5: tracking vs retention controlled by one forgetting factor lambda (VCL, paper 9.2).
- E6-A uses argmax of variance; for a GP with constant noise, A and B coincide for single queries.

- E3: independent z_t under the correct model; negative controls = wrong sigma_y and stale (post-update) variance; unknown-variance variant gives Student-t(2a_{t-1}).
- E4: hyperparameters fixed and shared (sf=1, ell=1, sigma_y=0.2); the PC-FSVI used is a minimal stand-in (inducing-point q(u), PC settling), whose fixed point is the conjugate update, so near-exact agreement with the GP is expected by construction. M<=5 inducing points breaks it (control).
- E4 latency: PC-FSVI is O(M^2), constant in t; exact rank-one GP is O(t^2), so the advantage over the exact GP appears only at large t (crossover ~ a few hundred steps in our run); vs batch refit O(t^3) it is immediate.
- E4-real: your code via adapter, same criteria. Streaming one point at a time is ~10x less accurate than 50-point chunks (1e-4 vs 7e-6 RMSE) because jitter accumulates per chained update. Latency 5.7 ms/update (constant) is dominated by 150 iterations that each re-invert M x M matrices; with lr=1.0 the update is exact Newton, and n_iters=1 gave identical accuracy at 0.10 ms in a side check (Gaussian likelihood only).
- E5-A: forgetting factor lambda and process noise are the same filter (var_pred = var/lambda = var + sigma_eta^2 when lambda_t = var/(var+sigma_eta^2)); a constant lambda* reproduces the Kalman variance only asymptotically (the pre-registered t>=100 tolerance was too tight; diff is 7e-9 by t>=150, a post-hoc diagnostic). Controls: static belief and mismatched sigma_eta (x0.2, x5) lose calibration. An EMA tuned to the steady-state Kalman gain ties the Kalman filter in RMSE and recovery; the Kalman advantage is calibrated uncertainty, the transient, and a principled gain. Untuned EMAs (alpha 0.02 / 0.5) are worse (RMSE 0.259 / 0.290 vs 0.154). The 1.96-sigma recovery metric can be gamed by inflated uncertainty (sigma_eta x5 'recovers' in 2 steps), so also report the fixed-band version.
- E5-A miss diagnosis (deterministic, longdouble): lambda* = P/(P+q) = 1-K_ss exactly; both variance maps share the fixed point P_ss; Kalman contracts at (1-K)^2=0.819 but constant-lambda at lambda*=0.905, so |diff| ~ lambda*^t (measured log-slope -0.0999 vs ln lambda* -0.1000); diff=1.0583e-6 at t=100, <1e-6 from t=101, 7.2e-9 at t=150. Ruled out: P_t vs P_{t|t-1}, lambda definition, float64 error (2.8e-17). Proposed v2 criteria (to be added beside, not replacing, the original): both converge to analytic P_ss (|Var(T=1000)-P_ss|<1e-12) and |diff| decays at rate lambda* (log-slope within 2% of ln lambda*).
- E5-B findings (stand-in, 5 seeds x 100 trials for B2/B3): kernel-correlated process noise gives a calibrated exact oracle (cov95 0.948-0.951, innovations N(0,1)); the stand-in matches it to 9e-6 (near-exact by construction). Tracking vs retention is real: raising sigma_eta (0 -> 0.02 -> 0.05 -> 0.1 -> 0.2) improves B2 recovery (inf -> 60 -> 23 -> 12 -> 8 steps) while the retention RMSE in the unobserved region grows (ratio 1.01 -> 1.37 -> 1.60 -> 1.54 -> 1.63); tracking error itself is best near sigma_eta 0.02-0.05. Multiplicative forgetting (S /= lambda) is UNSAFE for unobserved regions: lambda=0.9 recovers in 31 steps but retention RMSE explodes (0.25 -> 10.5) because variance grows like lambda^-t where no data arrives; forget toward the prior instead (additive noise, or Lambda <- lambda*Lambda + (1-lambda)*K^-1). Mapping to the real code: pc_infer's beta scales the prior precision (= lambda), and a drift time update would pass prior_cov = S + sigma_eta^2 K_zz.

## Open decisions
1. Does the predictive-coding layer only report errors (current), or drive the update as in PC-FSVI?
2. Contribution claim beyond Kalman filter / GP (needs to be stated before E4+).
3. Upload utils.py, uncertainty.py, metrics.py so the shim can be removed and continual_learning.py imports (needed for E5-B).
4. E8b feasibility: ellipsoidal sampling cost repeats every step; choose small d or a thinned schedule.

## Claims ladder
| After | What we may say |
|---|---|
| E1-E2 | scalar inference and sequential updating are exact and calibrated (Gaussian model) |
| E3 | prediction errors are correctly precision-weighted |
| E4-E4b | function-space belief matches the exact GP |
| E5-E6 | adaptation and uncertainty-driven action work in isolation |
| E7 | the closed loop works; ablations show which components matter |
| E8 | advanced inference preserves that behaviour |
