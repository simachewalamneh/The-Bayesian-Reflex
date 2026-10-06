## Bayesian Reflex: A Predictive Coding Engine

Unofficial implementation of [*The Bayesian Reflex: A Predictive Coding Engine for Artificial Intelligence*](https://arxiv.org/abs/2608.00492).

### Overview

The paper reinterprets the "Bayesian reflex" (online Bayesian learning) as a computational form of predictive coding. Its three mechanisms map onto free-energy minimisation:

1. **Belief maintenance:** a posterior over unknown parameters, held as particles, a GP posterior, an ellipsoidal decomposition, or a variational density.
2. **Sequential updating:** beliefs are revised with Bayes' theorem as new observations arrive.
3. **Uncertainty-driven action:** decisions use the full posterior to balance exploration and exploitation.

### Reference

- Paper: https://arxiv.org/abs/2608.00492

### Status

| Stage | Experiment | Status |
|---|---|---|
| E1 | Scalar Gaussian inference module + SBC calibration | validated (foundation only) |
| E2a | Sequential updating vs batch oracle (scalar Gaussian) | validated (foundation only) |
| E3 | One-step-ahead prediction error standardization (scalar Gaussian) | validated (foundation only) |
| E4 / E2b | Function space: exact GP oracle vs PC-FSVI **stand-in** (rank-one GP == batch GP) | validated for the stand-in only; real PC-FSVI not yet tested |
| E4-real | Same task with your real FSVI + pc_infer (adapter; utils.py shimmed) | validated in this setting (Gaussian, streaming); see docs |
| E5-A | Scalar drift tracking vs Kalman oracle (known linear-Gaussian model) | frozen: PASS under v2 (original v1 miss recorded and diagnosed) |
| E5-B | Functional drift, stand-in vs exact grid Kalman oracle | 20/21 checks; retention criterion missed; real PC-FSVI swap pending |
| E5-B-real | Functional drift with the real FSVI + pc_infer (same data as stand-in) | 19/20 checks; reproduces stand-in and oracle; one tolerance miss at beta=0.9 |
| E4b, E6-E8 |  function space, look-up table, nonstationarity, action, full loop, ellipsoidal | planned (see `docs/experimental_plan.md`) |

### Layout

```
src/bayesian_reflex/   GenerativeModel, ScalarGaussianBelief, ExactGP, PCFSVI (stand-in), PredictiveCodingLayer, BayesianReflex
src/bayesian_reflex/external/pc_fsvi/   your pc-fsvi modules (verbatim) + utils shim
experiments/           one script per stage (e1 implemented, e2..e8 stubs with protocol)
tests/                 unit tests
docs/                  experimental plan, decisions
results/               figures + json produced by experiments
```

### Quick start

```bash
pip install -r requirements.txt
pytest
python experiments/e1_scalar_belief.py
python experiments/e2_sequential_equivalence.py
python experiments/e3_prediction_error.py
python experiments/e4_function_space.py   # ~1-2 min
python experiments/e4_real_pcfsvi.py      # ~3-5 min
python experiments/e5a_scalar_drift.py
python experiments/e5b_functional_drift.py
python experiments/e5b_real_pcfsvi.py     # ~1-2 min
```

### Related papers

- Original chapter: https://arxiv.org/abs/2605.02825

