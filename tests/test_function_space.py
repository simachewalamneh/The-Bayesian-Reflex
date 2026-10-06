import numpy as np

from bayesian_reflex.models import ExactGP, PCFSVI, batch_gp


def _data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    return x, np.sin(x) + 0.2 * rng.standard_normal(n)


def test_sequential_gp_equals_batch():
    x, y = _data()
    gp = ExactGP(capacity=len(x))
    for xi, yi in zip(x, y):
        gp.update(yi, xi)
    xs = np.linspace(-3, 3, 50)
    (m1, v1), (m2, v2) = gp.predict_f(xs), batch_gp(x, y, xs)
    assert np.abs(m1 - m2).max() < 1e-9 and np.abs(v1 - v2).max() < 1e-9


def test_pc_settling_reaches_conjugate_fixed_point():
    x, y = _data()
    fs = PCFSVI(M=15)
    for xi, yi in zip(x, y):
        fs.update(yi, xi)
    m, S = fs.batch_posterior(x, y)
    assert np.abs(fs.m - m).max() < 1e-6 and np.abs(fs.S - S).max() < 1e-6
