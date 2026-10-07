# E7 engine confirmation protocol (declared BEFORE running; thresholds identical to the frozen E7 protocol)

Question: does the qualitative E7 conclusion survive replacing the exact-grid belief with (a) the PC-FSVI stand-in and (b) the user's real FSVI + pc_infer?
Scope: confirmation of the closed loop under engine substitution; one environment, one shift, one plasticity rule. Not the paper's LUT/ellipsoidal machinery.

## Design
- Same environment, agents (FULL, A1..A4, CONTROL), metrics and criteria C1..C6 as docs/e7_protocol.md; thresholds NOT changed.
- Reduced protocol (set from measured runtime, not from results): 5 seeds x 40 trials, T=200. Engines on the reduced protocol: exact-grid (apples-to-apples reference), stand-in, real.
- Additionally the stand-in runs the full 10 seeds x 200 trials (cheap, batched) for direct comparison with the E7 reference.
- In-loop agreement check (tests/test_engine_agents.py): engines track the exact agent to 5e-3 in z and means.

## Pre-declared handling of low power (no threshold is relaxed)
- C1, C2 (paired t-test, p<0.01): correct sign with 0.01 <= p < 0.10 -> INCONCLUSIVE; otherwise PASS/FAIL as specified.
- C4 (upper 95% CI < +5%): upper bound in [0.05, 0.10) -> INCONCLUSIVE.
- C3, C5, C6 and the control are point thresholds on pooled statistics: PASS/FAIL, never inconclusive.
- Engine verdict: PASS = all criteria PASS; INCONCLUSIVE = no FAIL but at least one INCONCLUSIVE; FAIL = any FAIL.
- "The conclusion survives engine substitution" is claimed only if the stand-in and real verdicts are both PASS. INCONCLUSIVE is reported as such, and is not rerun with more seeds unless declared as a new, separately labelled run.
