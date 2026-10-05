import numpy as np

from bayesian_reflex import BayesianReflex, PredictiveCodingLayer, ScalarGaussianBelief


def test_posterior_matches_closed_form_and_batch():
    rng = np.random.default_rng(0)
    mu0, s0, sy, T = 0.3, 1.2, 0.5, 40
    y = 1.5 + sy * rng.standard_normal(T)
    b = ScalarGaussianBelief(mu0, s0, sy, 1)
    for t in range(T):
        b.update(y[t:t + 1])
    prec = 1 / s0**2 + T / sy**2
    mean = (mu0 / s0**2 + y.sum() / sy**2) / prec
    assert np.allclose(b.mu, mean) and np.allclose(b.var, 1 / prec)


def test_standardized_errors_are_standard_normal():
    rng = np.random.default_rng(1)
    n, T, sy = 2000, 30, 0.5
    theta = rng.standard_normal(n)
    b = ScalarGaussianBelief(0.0, 1.0, sy, n)
    pc, z = PredictiveCodingLayer(), []
    for _ in range(T):
        y = theta + sy * rng.standard_normal(n)
        y_hat, v = b.predict()
        z.append(pc.standardized_error(y, y_hat, v))
        b.update(y)
    z = np.concatenate(z)
    assert abs(z.mean()) < 0.02 and abs(z.var() - 1) < 0.03


def test_reflex_step_keys_and_predict_before_update():
    b = ScalarGaussianBelief(0.0, 1.0, 0.5, 1)
    r = BayesianReflex(b, PredictiveCodingLayer(), policy=lambda m: float(m.sd[0]))
    out = r.step(np.array([1.0]))
    assert set(out) == {"y_hat", "var_pred", "z", "action"}
    assert out["y_hat"][0] == 0.0 and b.mu[0] != 0.0
