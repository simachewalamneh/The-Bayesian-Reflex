import numpy as np

from bayesian_reflex.models import ScalarDriftBelief, ScalarGaussianBelief


def _run(models, y):
    out = [[] for _ in models]
    for yt in y:
        for i, m in enumerate(models):
            m.update(np.array([yt])); out[i].append((m.mu[0], m.var[0]))
    return np.array(out)


def test_static_drift_equals_e1_and_matched_forgetting_equals_kalman():
    y = np.random.default_rng(0).standard_normal(100)
    a, b = ScalarDriftBelief(0, 1, 0.5, 0.0, 1), ScalarGaussianBelief(0, 1, 0.5, 1)
    ra = _run([a], y)[0]
    rb = []
    for yt in y:
        b.update(np.array([yt])); rb.append((b.mu[0], b.var[0]))
    assert np.abs(ra - np.array(rb)).max() < 1e-12
    k, f = ScalarDriftBelief(0, 1, 0.5, 0.05, 1), ScalarDriftBelief(0, 1, 0.5, 0.05, 1, forgetting="matched")
    r = _run([k, f], y)
    assert np.abs(r[0] - r[1]).max() < 1e-10
