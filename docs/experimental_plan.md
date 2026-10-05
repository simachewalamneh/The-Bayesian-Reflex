# Experimental plan (locked order)

Source: arXiv 2608.00492 (idea: Bayesian reflex as predictive coding) and arXiv 2605.02825 (original chapter: look-up table + ellipsoidal decomposition). Principle: every stage has an oracle or analytic reference.

| Stage | Question | Reference / oracle | Status |
|---|---|---|---|
| E1 | Scalar belief, accuracy + SBC calibration | analytic Gaussian posterior | **done (32/32 checks pass)** |
| E2 | Sequential = batch (Thm 2.1) | exact batch posterior | todo |
| E3 | Precision-weighted prediction error z_t ~ N(0,1) | exact predictive | todo |
| E4 | Function space: exact GP vs PC-FSVI | exact GP | todo |
| E4b | Look-up table principle (paper sec 5-6) | exact GP | todo |
| E5 | Nonstationarity: A scalar drift, B functional drift | Kalman filter | todo |
| E6 | Uncertainty-driven action: A variance, B info gain, C Thompson | random baseline / known regret | todo |
| E7 | Full reflex loop + ablations | composition of validated parts | todo |
| E8a | Ellipsoidal sampler standalone | exact samplers | todo |
| E8b | Ellipsoidal sampler inside the loop (Alg 4) | exact conjugate | todo |
| E8c | Recursive GP / online BNN (optional) | simpler inference | todo |

## Protocol rules
- Fix pass/fail criteria before coding; >=10 seeds; report mean +/- sd; every positive test has a negative control that must fail.
- E1-E3 are validation tests of the implementation; E4-E8 carry the research claims.

## Decisions recorded
- E1: coverage tolerance = 3 binomial SE; ranks from posterior samples (reusable for non-Gaussian methods).
- E3: z uses the one-step-ahead predictive, var = sigma_{t-1}^2 + sigma_y^2, computed before the update.
- E5: tracking vs retention controlled by one forgetting factor lambda (VCL, paper 9.2).
- E6-A uses argmax of variance; for a GP with constant noise, A and B coincide for single queries.

## Open decisions
1. Does the predictive-coding layer only report errors (current), or drive the update as in PC-FSVI?
2. Contribution claim beyond Kalman filter / GP (needs to be stated before E4+).
3. E8b feasibility: ellipsoidal sampling cost repeats every step; choose small d or a thinned schedule.
