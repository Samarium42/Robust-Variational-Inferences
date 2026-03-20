import numpy as np

from .utils import EPS, logsumexp, mixture_nll_from_logpk, project_to_simplex, responsibilities


def weights_uniform(K: int) -> np.ndarray:
    return np.ones(K, dtype=np.float64) / K


def weights_best_single(logpk: np.ndarray) -> np.ndarray:
    # choose component with lowest NLL alone on the same validation points
    K = logpk.shape[1]
    nlls = []
    for k in range(K):
        # mixture with weight 1 at k
        ll = logpk[:, k]
        nlls.append(float(-ll.mean()))
    best = int(np.argmin(nlls))
    w = np.zeros(K, dtype=np.float64)
    w[best] = 1.0
    return w


def grid_search_simplex(logpk: np.ndarray, step: float = 0.05) -> np.ndarray:
    # for small K only, but works fine for K up to 6 with moderate step
    K = logpk.shape[1]
    m = int(round(1.0 / step))
    if abs(m * step - 1.0) > 1e-9:
        raise ValueError("grid step must divide 1.0 exactly, for example 0.1, 0.05, 0.02")

    best_w = None
    best_val = 1e300

    def rec(idx: int, remaining: int, current):
        nonlocal best_w, best_val
        if idx == K - 1:
            counts = current + [remaining]
            w = np.array(counts, dtype=np.float64) / m
            val = mixture_nll_from_logpk(logpk, w)
            if val < best_val:
                best_val = val
                best_w = w
            return
        for c in range(remaining + 1):
            rec(idx + 1, remaining - c, current + [c])

    rec(0, m, [])
    return best_w


def em_weights(logpk: np.ndarray, iters: int = 200, tol: float = 1e-10) -> np.ndarray:
    K = logpk.shape[1]
    w = np.ones(K, dtype=np.float64) / K
    prev = mixture_nll_from_logpk(logpk, w)
    for _ in range(iters):
        r = responsibilities(logpk, w)
        w = r.mean(axis=0)
        w = np.clip(w, EPS, 1.0)
        w = w / w.sum()
        cur = mixture_nll_from_logpk(logpk, w)
        if abs(prev - cur) < tol:
            break
        prev = cur
    return w


def gradient_w(logpk: np.ndarray, w: np.ndarray) -> np.ndarray:
    # gradient of NLL wrt w, computed from responsibilities
    # nll = -mean log(sum w_k p_k)
    # d nll / d w_k = -mean (p_k / sum_j w_j p_j) = -mean (r_k / w_k)
    w = np.asarray(w, dtype=np.float64)
    w = np.clip(w, EPS, 1.0)
    w = w / w.sum()
    r = responsibilities(logpk, w)
    g = -(r / w[None, :]).mean(axis=0)
    return g.astype(np.float64)


def projected_gd(
    logpk: np.ndarray,
    iters: int = 400,
    lr: float = 0.05,
    momentum: float = 0.9,
) -> np.ndarray:
    K = logpk.shape[1]
    w = np.ones(K, dtype=np.float64) / K
    v = np.zeros(K, dtype=np.float64)
    for _ in range(iters):
        g = gradient_w(logpk, w)
        v = momentum * v + g
        w = project_to_simplex(w - lr * v)
    return w


def mirror_descent_exponentiated(
    logpk: np.ndarray,
    iters: int = 400,
    lr: float = 0.05,
) -> np.ndarray:
    K = logpk.shape[1]
    w = np.ones(K, dtype=np.float64) / K
    for _ in range(iters):
        g = gradient_w(logpk, w)
        w = w * np.exp(-lr * g)
        w = np.clip(w, EPS, 1.0)
        w = w / w.sum()
    return w


def frank_wolfe(
    logpk: np.ndarray,
    iters: int = 200,
    gamma_rule: str = "harmonic",
    line_search_points: int = 50,
) -> np.ndarray:
    # minimise NLL over simplex
    K = logpk.shape[1]
    w = np.ones(K, dtype=np.float64) / K

    for t in range(iters):
        g = gradient_w(logpk, w)  # gradient at w
        s = np.zeros(K, dtype=np.float64)
        s[int(np.argmin(g))] = 1.0  # best vertex for minimisation

        if gamma_rule == "harmonic":
            gamma = 2.0 / (t + 2.0)
        elif gamma_rule == "fixed":
            gamma = 0.1
        else:
            # simple grid line search on gamma
            best_gamma = 0.0
            best_val = mixture_nll_from_logpk(logpk, w)
            for j in range(1, line_search_points + 1):
                gg = j / line_search_points
                ww = (1 - gg) * w + gg * s
                val = mixture_nll_from_logpk(logpk, ww)
                if val < best_val:
                    best_val = val
                    best_gamma = gg
            gamma = best_gamma

        w = (1 - gamma) * w + gamma * s
        w = np.clip(w, EPS, 1.0)
        w = w / w.sum()

    return w


def coordinate_pair_search(
    logpk: np.ndarray,
    iters: int = 300,
    grid: int = 50,
    seed: int = 0,
) -> np.ndarray:
    # random pair coordinate descent with 1D search between two weights
    rng = np.random.default_rng(seed)
    K = logpk.shape[1]
    w = np.ones(K, dtype=np.float64) / K

    for _ in range(iters):
        a, b = rng.choice(K, size=2, replace=False)
        mass = w[a] + w[b]
        if mass <= 0:
            continue

        best = None
        best_val = 1e300
        for j in range(grid + 1):
            t = j / grid
            w_try = w.copy()
            w_try[a] = t * mass
            w_try[b] = (1 - t) * mass
            val = mixture_nll_from_logpk(logpk, w_try)
            if val < best_val:
                best_val = val
                best = w_try

        w = best
        w = np.clip(w, EPS, 1.0)
        w = w / w.sum()

    return w