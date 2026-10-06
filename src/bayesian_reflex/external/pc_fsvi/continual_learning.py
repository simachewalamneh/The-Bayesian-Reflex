import numpy as np
from scipy.optimize import minimize as spminimize

from .variational import FSVI
from .predictive_coding import pc_infer
from .gp import ExactGP
from .uncertainty import standardized_error
from .utils import rbf_kernel
from . import metrics as M


def _safe_chol(S, jitter=1e-8):
    """Symmetrise and add escalating jitter so near-singular covariances
    (e.g. beta=0 with a rank-deficient A^T A) do not crash Cholesky."""
    S = 0.5 * (S + S.T)
    for k in range(12):
        try:
            return np.linalg.cholesky(S + jitter * (10.0 ** k) * np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("covariance not PD even with jitter")


def _score(predict_fn, j, task_data, eval_data):
    """RMSE of predict_fn on task j. Uses the held-out grid if eval_data is
    given, otherwise falls back to the task's training inputs."""
    if eval_data is not None:
        Xj, fj = eval_data[j]
    else:
        Xj, _, fj = task_data[j]
    mean, _ = predict_fn(Xj)
    return M.rmse(fj, mean)


def independent_baseline(task_data, Z, kernel_kwargs, eval_data=None, maxiter=3000):
    """Lower bound: a FRESH model per task, no memory. Row i = model_i scored
    on tasks 0..i (so off-diagonals show pure catastrophic forgetting)."""
    T = len(task_data)
    R = np.full((T, T), np.nan)
    for i, (Xi, yi, _) in enumerate(task_data):
        model = FSVI(Z, **kernel_kwargs)
        model.fit(Xi, yi, beta=1.0, prior_mean=None, prior_cov=None, maxiter=maxiter)
        for j in range(i + 1):
            R[i, j] = _score(model.predict, j, task_data, eval_data)
    return R


def sequential_gp_baseline(task_data, eval_data=None, noise_std=0.1,
                           lengthscale=1.0, variance=1.0):
    """Exact GP refit on all data seen so far = exact Bayes for a SINGLE-function
    model. Not an 'oracle' for CL when tasks conflict."""
    T = len(task_data)
    R = np.full((T, T), np.nan)
    X_all, y_all = None, None
    for i, (Xi, yi, _) in enumerate(task_data):
        X_all = Xi if X_all is None else np.vstack([X_all, Xi])
        y_all = yi if y_all is None else np.concatenate([y_all, yi])
        gp = ExactGP(lengthscale=lengthscale, variance=variance, noise_std=noise_std)
        gp.fit(X_all, y_all)
        for j in range(i + 1):
            R[i, j] = _score(gp.predict, j, task_data, eval_data)
    return R


def fsvi_continual(task_data, Z, kernel_kwargs, beta=1.0, eval_data=None, maxiter=3000):
    T = len(task_data)
    R = np.full((T, T), np.nan)
    model = FSVI(Z, **kernel_kwargs)
    prior_mean, prior_cov = None, None
    for i, (Xi, yi, _) in enumerate(task_data):
        model.fit(Xi, yi, beta=beta, prior_mean=prior_mean, prior_cov=prior_cov, maxiter=maxiter)
        prior_mean, prior_cov = model.posterior()
        for j in range(i + 1):
            R[i, j] = _score(model.predict, j, task_data, eval_data)
    return R, model


def surprise(model, X, y):
    """mean z^2 of new labels under the model's current posterior (~1 if consistent)."""
    mean, var = model.predict(X)
    z = standardized_error(y, mean, var, model.noise_std)
    return float(np.mean(z ** 2))


def surprise_beta(i, model, X, y, lo=0.05, hi=5.0):
    """Adaptive stability weight from PREDICTIVE SURPRISE on the new task's
    labels, computed under the current posterior BEFORE fitting:
        s = mean z^2,  z = (y - mean) / sqrt(var_f + sigma_n^2)
        beta = clip(1 / s, lo, hi)
    s ~ 1  -> data consistent with the old posterior -> beta ~ 1 (stable)
    s >> 1 -> task conflicts with what is known      -> beta small (plastic)
    Input-space variance alone cannot do this: when tasks share inputs the
    variance is low even though the function changed (concept shift)."""
    if i == 0:
        return 1.0
    s = surprise(model, X, y)
    return float(np.clip(1.0 / max(s, 1e-8), lo, hi))


def pc_fsvi_continual(
    task_data, Z, kernel_kwargs, beta=1.0, n_iters=150, lr=1.0,
    adaptive_precision=False, precision_override=None, eval_data=None,
):
    """beta: float, or callable beta(i, model, X, y) -> float (per-task)."""
    T = len(task_data)
    R = np.full((T, T), np.nan)
    model = FSVI(Z, **kernel_kwargs)
    prior_mean = np.zeros(model.M)
    prior_cov = model.Kzz.copy()
    histories, betas = [], []
    for i, (Xi, yi, _) in enumerate(task_data):
        b = beta(i, model, Xi, yi) if callable(beta) else float(beta)
        betas.append(b)
        m, S, hist = pc_infer(
            model, Xi, yi, prior_mean, prior_cov, beta=b, n_iters=n_iters, lr=lr,
            adaptive_precision=adaptive_precision, precision_override=precision_override,
        )
        model.m, model.L = m, _safe_chol(S)
        hist["beta"] = b
        histories.append(hist)
        prior_mean, prior_cov = model.posterior()
        for j in range(i + 1):
            R[i, j] = _score(model.predict, j, task_data, eval_data)
    model.betas_used = betas
    return R, model, histories


def ewc_like_baseline(task_data, Z, kernel_kwargs, lam=10.0, maxiter=3000,
                      accumulate=True, eval_data=None):
    """Point-estimate EWC on the inducing values u.
    accumulate=True : online EWC, Fisher summed over tasks (protects ALL past tasks)
    accumulate=False: old behaviour, Fisher overwritten (only protects the last task)"""
    T = len(task_data)
    R = np.full((T, T), np.nan)
    model = FSVI(Z, **kernel_kwargs)
    m_star = np.zeros(model.M)
    Fisher = np.zeros(model.M)
    sigma_n2 = model.noise_std ** 2

    for i, (Xi, yi, _) in enumerate(task_data):
        A = model._Kxz(Xi) @ model.Kzz_inv

        def neg_log_post(m, A=A, yi=yi, Fisher=Fisher, m_star=m_star):
            resid = yi - A @ m
            gp_prior = 0.5 * m @ model.Kzz_inv @ m  # keeps u well-posed where no data was seen
            return (0.5 * np.sum(resid ** 2) / sigma_n2 + gp_prior
                    + 0.5 * lam * np.sum(Fisher * (m - m_star) ** 2))

        res = spminimize(neg_log_post, m_star.copy(), method="L-BFGS-B", options={"maxiter": maxiter})
        m_star = res.x
        model.m = m_star
        model.L = np.eye(model.M) * 1e-3
        F_new = np.sum(A ** 2, axis=0) / sigma_n2
        Fisher = Fisher + F_new if accumulate else F_new
        for j in range(i + 1):
            R[i, j] = _score(model.predict, j, task_data, eval_data)
    return R, model


def log_evidence(model, X, y, m, S):
    """log p(y | X) under q(u)=N(m,S) and the GP conditional + Gaussian noise:
        y ~ N(A m,  Kxx - A Kzx + A S A^T + sigma_n^2 I),  A = Kxz Kzz^-1.
    With (m,S)=(0,Kzz) this is the fresh-GP-prior evidence."""
    Kxz = model._Kxz(X)
    A = Kxz @ model.Kzz_inv
    Kxx = rbf_kernel(X, X, model.lengthscale, model.kernel_variance)
    C = Kxx - A @ Kxz.T + A @ S @ A.T + model.noise_std ** 2 * np.eye(len(X))
    L = _safe_chol(C)
    r = np.linalg.solve(L, np.asarray(y) - A @ m)
    return float(-0.5 * r @ r - np.sum(np.log(np.diag(L))) - 0.5 * len(X) * np.log(2 * np.pi))


def spawn_continual(task_data, Z, kernel_kwargs, thr=3.0, beta=1.0,
                    n_iters=150, lr=1.0, eval_data=None,
                    criterion="surprise", margin=0.0):
    """Detect-and-spawn. Keep a list of FSVI components (each = its own q(u)).

    criterion="surprise": s = mean z^2 under each component;
        merge into argmin-s component if s < thr, else spawn.
        thr=0 -> ALWAYS spawn; thr=inf -> ALWAYS merge (= plain chaining).
    criterion="evidence": Bayes-factor test, threshold-free. Compare
        log p(y|X) under the best existing component vs under a fresh GP prior;
        merge if  log_ev_best + margin >= log_ev_fresh, else spawn.

    Merge -> chain (posterior becomes prior; exact Bayes). Spawn -> fresh
    component from the GP prior; old components untouched.
    Evaluation is task-incremental (task j predicted by its assigned component).
    Returns (R, comps, assign)."""
    T = len(task_data)
    R = np.full((T, T), np.nan)
    comps, assign = [], []
    for i, (Xi, yi, _) in enumerate(task_data):
        merge, k = False, -1
        if comps:
            if criterion == "surprise":
                sc = [surprise(c, Xi, yi) for c in comps]
                k = int(np.argmin(sc))
                merge = sc[k] < thr
            elif criterion == "evidence":
                ev = [log_evidence(c, Xi, yi, *c.posterior()) for c in comps]
                k = int(np.argmax(ev))
                fresh = FSVI(Z, **kernel_kwargs)
                ev_new = log_evidence(fresh, Xi, yi, np.zeros(fresh.M), fresh.Kzz)
                merge = ev[k] + margin >= ev_new
            else:
                raise ValueError(criterion)
        if merge:
            model = comps[k]
            prior_mean, prior_cov = model.posterior()
        else:
            model = FSVI(Z, **kernel_kwargs)
            comps.append(model)
            k = len(comps) - 1
            prior_mean, prior_cov = np.zeros(model.M), model.Kzz.copy()
        m, S, _ = pc_infer(model, Xi, yi, prior_mean, prior_cov, beta=beta,
                           n_iters=n_iters, lr=lr)
        model.m, model.L = m, _safe_chol(S)
        assign.append(k)
        for j in range(i + 1):
            R[i, j] = _score(comps[assign[j]].predict, j, task_data, eval_data)
    return R, comps, assign
