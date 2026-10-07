import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
import e5b_functional_drift as e5b  # noqa: E402
from bayesian_reflex.models.grid_reflex import GridReflexAgent, InducingGridAgent, RealGridAgent  # noqa: E402


def _drive(agent, F, eps, ridx):
    zs = []
    for t in range(len(F)):
        _, z = agent.step(F[t], eps[t], ridx[t]); zs.append(z)
    return np.array(zs)


def test_engines_track_the_exact_grid_agent_in_the_loop():
    rng = np.random.default_rng(0)
    n, T, G = 3, 40, e5b.G
    f = (e5b.LK @ rng.standard_normal((G, n))).T
    F = []
    for _ in range(T):
        f = f + e5b.SETA * (e5b.LK @ rng.standard_normal((G, n))).T; F.append(f.copy())
    F, eps, ridx = np.array(F), rng.standard_normal((T, n)), rng.integers(0, G, (T, n))
    mk = dict(sy_true=e5b.SY, seta=e5b.SETA, policy="random", plastic=True)
    ex = GridReflexAgent(n, e5b.K, **mk)
    st = InducingGridAgent(n, grid=e5b.GRID, **mk)
    re = RealGridAgent(n, grid=e5b.GRID, **mk)
    zx, zs, zr = _drive(ex, F, eps, ridx), _drive(st, F, eps, ridx), _drive(re, F, eps, ridx)
    assert np.abs(zx - zs).max() < 5e-3 and np.abs(zx - zr).max() < 5e-3
    assert np.abs(ex.m - st.mean_sd()[0]).max() < 5e-3 and np.abs(ex.m - re.mean_sd()[0]).max() < 5e-3
