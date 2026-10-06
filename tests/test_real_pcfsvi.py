import numpy as np

from bayesian_reflex.external.pc_fsvi import closed_form_optimum
from bayesian_reflex.models import batch_gp
from bayesian_reflex.models.pcfsvi_adapter import RealPCFSVI


def test_real_pcfsvi_streaming_matches_exact_gp_and_closed_form():
    rng = np.random.default_rng(0)
    x = rng.uniform(-3, 3, 60)
    y = np.sin(x) + 0.2 * rng.standard_normal(60)
    r = RealPCFSVI(M=20, n_iters=5, lr=1.0)
    for xi, yi in zip(x, y):
        r.update(yi, xi)
    xs = np.linspace(-3, 3, 100)
    mu, v = r.predict_f(xs)
    mg, vg = batch_gp(x, y, xs, sy=0.2)
    assert np.sqrt(np.mean((mu - mg) ** 2)) < 1e-3
    m, S = closed_form_optimum(r.model, x.reshape(-1, 1), y, np.zeros(20), r.model.Kzz, 1.0)
    assert np.abs(r.m - m).max() < 1e-3
