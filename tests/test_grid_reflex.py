import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
import e5b_functional_drift as e5b  # noqa: E402
from bayesian_reflex.models.grid_reflex import GridReflexAgent  # noqa: E402


def test_agent_without_plasticity_equals_grid_kalman_and_variance_policy_is_argmax():
    rng = np.random.default_rng(0)
    G, T = e5b.G, 40
    idx = rng.integers(0, G, T)
    f = (e5b.LK @ rng.standard_normal((G, 1))).T
    ag = GridReflexAgent(1, e5b.K, e5b.SY, e5b.SETA, policy="random", plastic=False)
    kal = e5b.GridKalman(e5b.SETA, 1)
    for t in range(T):
        f = f + e5b.SETA * (e5b.LK @ rng.standard_normal((G, 1))).T
        eps = rng.standard_normal(1)
        _, z = ag.step(f, eps, np.array([idx[t]]))
        zk = kal.step(idx[t], f[:, idx[t]] + e5b.SY * eps)
        assert abs(z[0] - zk[0]) < 1e-9
    assert np.abs(ag.m - kal.m).max() < 1e-9
    v = GridReflexAgent(1, e5b.K, e5b.SY, e5b.SETA, policy="variance", plastic=False)
    i, _ = v.step(f, rng.standard_normal(1), np.array([0]))
    assert i[0] == int(np.argmax(np.diag(e5b.K + e5b.SETA ** 2 * e5b.K)))
