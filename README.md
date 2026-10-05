## Bayesian Reflex: A Predictive Coding Engine

Unofficial implementation of [*The Bayesian Reflex: A Predictive Coding Engine for Artificial Intelligence*](https://arxiv.org/abs/2608.00492).

### Overview

The paper reinterprets the "Bayesian reflex" (online Bayesian learning) as a computational form of predictive coding. Its three mechanisms map onto free-energy minimisation:

1. **Belief maintenance:** a posterior over unknown parameters, held as particles, a GP posterior, an ellipsoidal decomposition, or a variational density.
2. **Sequential updating:** beliefs are revised with Bayes' theorem as new observations arrive.
3. **Uncertainty-driven action:** decisions use the full posterior to balance exploration and exploitation.

### Reference

- Paper: https://arxiv.org/abs/2608.00492