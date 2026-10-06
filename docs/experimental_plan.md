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
| E5 | Nonstationarity: A scalar drift, B functional drift | Kalman filter | todo |
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
