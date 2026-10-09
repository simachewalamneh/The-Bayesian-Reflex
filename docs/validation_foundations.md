# Foundations validation: E1-E3

Scope: these three experiments validate the scalar Gaussian inference module of the Bayesian-reflex implementation (belief maintenance, sequential updating, standardized one-step-ahead prediction error). They do NOT test the reflex loop, function-space inference, nonstationarity, action, or the paper-specific machinery; those come in later PRs.

| Stage | Question | Reference | Result |
|---|---|---|---|
| E1 | Is the scalar belief accurate and calibrated (simulation-based calibration, 1000 trials x 10 seeds)? | analytic Gaussian posterior | 32/32 checks |
| E2a | Does the recursive update equal the batch posterior (T=100, stress T=1e5)? | batch posterior in extended precision | 9/9 checks |
| E3 | Are one-step-ahead errors correctly standardized by the predictive variance? Kalman identity, controls, unknown-variance (Student-t) variant | exact predictive | 29/29 checks |

Method rules used throughout: pass/fail criteria are fixed before running; every positive test has a negative control that must fail (wrong noise level, stale variance, dropped data, wrong CDF); >= 10 seeds with confidence intervals; claims are limited to what a stage tests.

Run (about 1 minute in total):
    pip install -r requirements.txt
    pytest
    python experiments/e1_scalar_belief.py
    python experiments/e2_sequential_equivalence.py
    python experiments/e3_prediction_error.py

Layout: src/bayesian_reflex/ (GenerativeModel, ScalarGaussianBelief, PredictiveCodingLayer, BayesianReflex), experiments/, tests/, results/ (figures and json written by the scripts).
