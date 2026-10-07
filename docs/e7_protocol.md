# E7 protocol (DRAFT, awaiting approval): the closed Bayesian reflex loop with ablations

Scope if it passes: the integrated loop (belief with drift model -> prediction + standardized error -> belief update -> uncertainty-driven action -> new observation) behaves coherently in a drifting, abruptly shifting environment, and each component earns its place by ablation. Built on our validated components (exact functional Kalman belief; the stand-in and real PC-FSVI afterwards). It does NOT test the paper's look-up-table or ellipsoidal machinery (E4b/E8 are deferred by decision) and does not validate the paper's claims about them.

## Loop (paper Fig. 1, our components)
Observe y_t at the chosen x_t -> belief predicts (mean, var) before seeing y_t -> PC layer standardizes z_t = (y_t - mean)/sqrt(var) -> Bayes update with drift (process noise) -> uncertainty -> policy chooses x_{t+1} (variance sampling) -> environment.

## Environment
f_0 ~ GP(0, K_RBF); f_t = f_{t-1} + eta_t, eta ~ N(0, 0.05^2 K); abrupt shift at t=100: f <- f + (cos x - sin x); sigma_y = 0.2; grid/candidates G=41 on [-3,3]; T=200; 200 trials x 10 seeds (seed = block of trials).

## Agents (exact grid Kalman belief; same engine, one factor removed at a time)
- FULL: drift model sigma_eta=0.05 + variance-driven action + error-driven plasticity: q_t = sigma_eta^2 * clip(s_t, 1, 100), s_t = mean of z^2 over the last 5 steps (computed before the update).
- A1 no action (random queries). A2 no drift model (sigma_eta=0, static belief). A3 no PC feedback (fixed sigma_eta, no plasticity). A4 none of the three.
- CONTROL: FULL with sigma_y assumed 0.5x the true value.

## Metrics
Grid RMSE over time; recovery steps after the shift (first step with RMSE <= 1.5x the agent's own pre-shift mean RMSE, held 5 steps); 95% coverage; mean z^2 around the shift.

## Pre-registered criteria
- C1 action matters: FULL mean RMSE (t=1..200) < A1, paired over seeds, p<0.01.
- C2 drift model matters: FULL < A2 (same test).
- C3 PC feedback speeds recovery: median recovery FULL <= 0.8 x A3.
- C4 PC feedback is harmless pre-shift: upper 95% CI bound of (RMSE_FULL - RMSE_A3)/RMSE_A3 over t=20..99 < +5%.
- C5 calibration pre-shift (t=20..99): |cov95 - 0.95| < 0.03 for FULL.
- C6 the error signal detects the shift: for A3, mean z^2 over t=101..105 > 3, and mean z^2 over t=20..99 within 3 SE of 1.
- CONTROL: wrong-sigma_y FULL has mean z^2 over t=20..99 > 1.5 (must fail calibration).
- INFO: FULL vs A4 total effect; interaction between action and drift model; behaviour with the stand-in and real engines.

## Implementation note
Error-driven plasticity makes the covariance differ per trial, so the belief is batched (n x G x G); the cost is acceptable at G=41.

## Decisions needed
1. PC layer role: the plasticity rule above makes the error DRIVE the update (resolving the open decision), and is itself a new, unvalidated mechanism tested by C3/C4. Approve, or restrict E7 to report-only (drop C3/C4/A3)?
2. Engine order: exact grid first, then stand-in and real as confirmation.
3. Confirm that E4b is deferred and that claims stay scoped as above.
4. Approve the thresholds now, not after the results.
