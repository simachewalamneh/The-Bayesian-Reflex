import numpy as np

from bayesian_reflex.models.lookup_table import (LookupTableEmulator, exact_chol_trajectory, exact_naive_trajectory,
                                                gauss_kernel, kl_gauss)

XD = np.linspace(-3, 3, 8)[:, None]
YD = 0.6 * np.sin(XD[:, 0]) + 0.4
grid = lambda N: (-4 + (np.arange(N) + 0.5) * 8.0 / N)[:, None]
V = np.linspace(-2.63, 2.71, 6)[:, None]


def test_lut_joint_mean_equals_exact_mean_and_kl_shrinks_with_grid():
    kl = []
    for N in (5, 20, 80):
        emu = LookupTableEmulator(XD, YD, grid(N), gauss_kernel, jitter=1e-12)
        m_ex, C_ex = emu.joint_exact(V)
        m_l, C_l = emu.joint_lut(V)
        assert np.abs(m_ex - m_l).max() < 1e-6
        kl.append(kl_gauss(m_ex, C_ex, m_l, C_l, jit=1e-12))
    assert kl[0] > kl[1] > kl[2] and kl[0] > 1e-3 and kl[2] < 1e-9   # with jitter 1e-12; at jitter 1e-8 the KL is jitter-limited (~2.6e-5 at N=80)


def test_simulation_matches_analytic_lut_joint():
    emu = LookupTableEmulator(XD, YD, grid(60), gauss_kernel)
    m, C = emu.joint_lut(V[:4])
    x, ff = emu.simulate(40000, 4, np.random.default_rng(0), lambda t, yl: np.tile(V[t - 1], (len(yl), 1)))
    assert np.isinf(ff).all()
    se = np.sqrt(np.diag(C) / 40000)
    assert (np.abs(x.mean(0) - m) / se).max() < 5
    assert np.abs(np.cov(x.T) - C).max() < 0.02


def test_exact_variants_agree_early_and_naive_fails_on_clustered_inputs():
    f = lambda t, yl: np.array([[yl[0]]])
    xc, fc = exact_chol_trajectory(XD, YD, gauss_kernel, -2.0, 12, f, np.random.default_rng(1), jitter=1e-8)
    xn, fn = exact_naive_trajectory(XD, YD, gauss_kernel, -2.0, 12, f, np.random.default_rng(1))
    assert np.isinf(fc) and np.abs(xc[:5] - xn[:5]).max() < 1e-3      # agree while the inputs are still separated
    assert np.isfinite(fn)                                              # explicit inverse, no jitter: breaks down once inputs cluster
