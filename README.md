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
| E2b-E8 | GP sequential equivalence, prediction error, function space, look-up table, nonstationarity, action, full loop, ellipsoidal | planned (see `docs/experimental_plan.md`) |

### Layout

```
src/bayesian_reflex/   GenerativeModel, ScalarGaussianBelief, PredictiveCodingLayer, BayesianReflex
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
```

### Related papers

- Original chapter: https://arxiv.org/abs/2605.02825

