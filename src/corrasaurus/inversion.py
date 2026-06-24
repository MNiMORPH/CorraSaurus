"""Joint inversion for per-lithology attrition distances.

Given observed clast fractions at many sample sites and the upstream source-cell
distributions (one ``SourceCells`` object), solve for the vector of per-lithology
e-folding distances ``l_k`` that best reproduces the observations across *all*
sites at once.

Two estimators on the fraction data:

* :func:`invert` -- parameterises the unknowns as ``theta = log(l_k)`` (so the
  optimiser is unconstrained while ``l_k`` stays positive) and minimises either a
  count-weighted least-squares (``"lsq"``) or a negative multinomial
  log-likelihood (``"multinomial"``) objective.  The original, simple estimator.
* :func:`invert_alpha` -- reparameterises in ``alpha = 1/l`` so the durable limit
  is the well-defined ``alpha = 0`` (no artificial bound), with an analytic
  Jacobian and Gauss-Newton uncertainties.  :func:`invert_alpha_coorddescent` is
  a coordinate-descent cross-check on it.

For the modular process x data-channel model (abrasion / fragmentation /
production over mass, count, and clast-size observations), see
:mod:`corrasaurus.attrition`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, least_squares, minimize_scalar

from .model import SourceCells, predicted_fractions, counts_and_moments


@dataclass
class InversionResult:
    l_k: np.ndarray          # fitted attrition distances [m], canonical order
    success: bool
    objective: str
    cost: float              # final objective value
    n_sites_used: int        # sites with both a prediction and observations
    raw: object              # the underlying scipy OptimizeResult


def _prepare(
    f_obs: np.ndarray,
    counts_total: np.ndarray | None,
    cells: SourceCells,
):
    """Validate shapes and build the site mask (sites that can be predicted)."""
    f_obs = np.asarray(f_obs, dtype=float)
    if f_obs.shape != (cells.n_sites, cells.n_liths):
        raise ValueError(
            f"f_obs shape {f_obs.shape} != ({cells.n_sites}, {cells.n_liths})"
        )
    if counts_total is None:
        counts_total = np.ones(cells.n_sites)
    counts_total = np.asarray(counts_total, dtype=float)

    # A site is usable only if it has observations and at least one upstream
    # source cell (so the forward model can predict something).
    has_source = np.zeros(cells.n_sites, dtype=bool)
    has_source[np.unique(cells.site_idx)] = True
    has_obs = np.isfinite(f_obs).all(axis=1) & (f_obs.sum(axis=1) > 0)
    mask = has_source & has_obs
    return f_obs, counts_total, mask


def _objective_lsq(theta, f_obs, counts_total, cells, mask):
    f_pred = predicted_fractions(np.exp(theta), cells)
    resid = (f_pred - f_obs)[mask]
    w = counts_total[mask][:, None]
    return float(np.sum(w * resid**2))


def _objective_multinomial(theta, f_obs, counts_total, cells, mask):
    f_pred = predicted_fractions(np.exp(theta), cells)
    eps = 1e-12
    p = np.clip(f_pred[mask], eps, 1.0)
    n_obs = counts_total[mask][:, None] * f_obs[mask]  # reconstruct counts
    return float(-np.sum(n_obs * np.log(p)))


_OBJECTIVES = {"lsq": _objective_lsq, "multinomial": _objective_multinomial}


def invert(
    f_obs: np.ndarray,
    cells: SourceCells,
    counts_total: np.ndarray | None = None,
    l0: np.ndarray | float = 20_000.0,
    objective: str = "lsq",
    bounds_m: tuple[float, float] = (100.0, 1.0e7),
) -> InversionResult:
    """Jointly invert for per-lithology attrition distances.

    Parameters
    ----------
    f_obs : array (n_sites, n_liths)
        Observed clast-count fractions in canonical lithology order.
    cells : SourceCells
        Upstream source-cell distribution for every site.
    counts_total : array (n_sites,), optional
        Number of clasts counted per site (least-squares / multinomial weight).
        Defaults to equal weight.
    l0 : float or array (n_liths,)
        Initial guess for the attrition distances [m].
    objective : {"lsq", "multinomial"}
    bounds_m : (lo, hi)
        Hard bounds on each ``l_k`` [m]; keeps the optimiser in a sane range.
    """
    if objective not in _OBJECTIVES:
        raise ValueError(f"unknown objective {objective!r}; choose from {list(_OBJECTIVES)}")
    obj = _OBJECTIVES[objective]
    f_obs, counts_total, mask = _prepare(f_obs, counts_total, cells)
    if mask.sum() == 0:
        raise ValueError("no usable sites (need both observations and upstream sources)")

    l0 = np.broadcast_to(np.asarray(l0, dtype=float), (cells.n_liths,)).copy()
    theta0 = np.log(l0)
    theta_bounds = [(np.log(bounds_m[0]), np.log(bounds_m[1]))] * cells.n_liths

    res = minimize(
        obj, theta0, args=(f_obs, counts_total, cells, mask),
        method="L-BFGS-B", bounds=theta_bounds,
    )
    return InversionResult(
        l_k=np.exp(res.x),
        success=bool(res.success),
        objective=objective,
        cost=float(res.fun),
        n_sites_used=int(mask.sum()),
        raw=res,
    )


# ---------------------------------------------------------------------------
# alpha = 1/l parameterisation: analytic Jacobian, Levenberg-Marquardt,
# and Hessian-based uncertainty.  alpha = 0 is the well-defined "infinitely
# durable" limit, so durable lithologies no longer pin at an artificial l bound.
# ---------------------------------------------------------------------------

@dataclass
class AlphaResult:
    alpha: np.ndarray          # fitted attrition coefficients [1/m]
    alpha_std: np.ndarray      # 1-sigma from the Gauss-Newton Hessian [1/m]
    l_k: np.ndarray            # 1/alpha [m] (inf where alpha == 0)
    l_lower_km: np.ndarray     # one-sided lower bound on l* [km] = 1/(alpha+sigma)
    at_bound: np.ndarray       # bool: alpha pinned at 0 (durable, lower-bounded)
    cost: float
    success: bool
    n_sites_used: int
    raw: object


def _resid_jac(alpha, f_obs, w, cells, mask):
    """Weighted fraction residuals and their analytic Jacobian (d resid/d alpha).

    df_k/dalpha_m = f_k * e_m - delta_km * e_k,  where e_m = D_m / S, and
    S = sum_m N_m.  See model.counts_and_moments for N, D.
    """
    N, D = counts_and_moments(alpha, cells)
    S = N.sum(axis=1, keepdims=True)
    f = N / S
    e = D / S                                   # (n_sites, n_liths)
    sw = np.sqrt(w[mask])                        # (ns,)
    fm, em = f[mask], e[mask]                     # (ns, nl)
    resid = (sw[:, None] * (fm - f_obs[mask])).ravel()
    # Jacobian: J[s,k,m] = sw_s * (f_k e_m - delta_km e_k)
    nl = cells.n_liths
    J = fm[:, :, None] * em[:, None, :]           # f_k e_m
    diag = np.arange(nl)
    J[:, diag, diag] -= em                        # subtract e_k on the diagonal
    J *= sw[:, None, None]
    return resid, J.reshape(-1, nl)


def invert_alpha(
    f_obs: np.ndarray,
    cells: SourceCells,
    counts_total: np.ndarray | None = None,
    l0: float = 20_000.0,
    l_min_m: float = 100.0,
) -> AlphaResult:
    """Joint inversion in alpha = 1/l, with uncertainty from the GN Hessian.

    alpha is bounded to [0, 1/l_min_m]; alpha = 0 means perfectly durable.
    Count-weighted least squares on the raw fractions, with the analytic
    Jacobian from :func:`_resid_jac`.
    """
    f_obs, counts_total, mask = _prepare(f_obs, counts_total, cells)
    if mask.sum() == 0:
        raise ValueError("no usable sites")
    nl = cells.n_liths
    # Optimise in u = alpha [1/km] = 1000 * alpha[1/m] so the variable is O(1)
    # (alpha in 1/m is ~1e-5, below the optimisers' tolerances).  l*[km] = 1/u.
    SC = 1.0e3
    u_max = SC / l_min_m
    u0 = np.full(nl, SC / l0)

    res = least_squares(
        lambda u: _resid_jac(u / SC, f_obs, counts_total, cells, mask)[0],
        u0, jac=lambda u: _resid_jac(u / SC, f_obs, counts_total, cells, mask)[1] / SC,
        bounds=(0.0, u_max), method="trf",
    )

    # Gauss-Newton covariance (in u-space) from the Jacobian at the solution.
    J = res.jac
    m, n = J.shape
    dof = max(m - n, 1)
    s2 = 2.0 * res.cost / dof
    u_std = np.sqrt(np.clip(np.diag(np.linalg.pinv(J.T @ J) * s2), 0, None))

    u = res.x
    alpha = u / SC
    alpha_std = u_std / SC
    at_bound = u <= 1e-9
    with np.errstate(divide="ignore"):
        l_k = np.where(alpha > 0, 1.0 / alpha, np.inf)
    l_lower_km = 1.0 / (u + u_std)                 # one-sided lower bound on l* [km]

    return AlphaResult(
        alpha=alpha, alpha_std=alpha_std, l_k=l_k, l_lower_km=l_lower_km,
        at_bound=at_bound, cost=float(res.cost), success=bool(res.success),
        n_sites_used=int(mask.sum()), raw=res,
    )


def invert_alpha_coorddescent(
    f_obs, cells, counts_total=None, l0=20_000.0, l_min_m=100.0,
    max_sweeps=50, tol=1e-10,
):
    """Coordinate descent in alpha: cycle 1-D solves, one lithology at a time.

    Exploits the separability (N_k depends only on alpha_k; lithologies couple
    only through the normalisation) and the monotone/convex Laplace structure,
    so each 1-D subproblem is unimodal.  A robust cross-check on invert_alpha.
    """
    f_obs, counts_total, mask = _prepare(f_obs, counts_total, cells)
    nl = cells.n_liths
    SC = 1.0e3                       # optimise in u = alpha[1/km]; l*[km] = 1/u
    u_max = SC / l_min_m
    u = np.full(nl, SC / l0)
    w = counts_total[mask][:, None]

    def cost_full(uu):
        N, _ = counts_and_moments(uu / SC, cells)
        f = N / N.sum(axis=1, keepdims=True)
        return float(np.sum(w * (f[mask] - f_obs[mask]) ** 2))

    prev = cost_full(u)
    for _ in range(max_sweeps):
        for k in range(nl):
            def cost_k(uk, k=k):
                uu = u.copy(); uu[k] = uk
                return cost_full(uu)
            r = minimize_scalar(cost_k, bounds=(0.0, u_max), method="bounded",
                                options={"xatol": 1e-8})
            u[k] = r.x
        cur = cost_full(u)
        if abs(prev - cur) < tol * max(prev, 1e-30):
            break
        prev = cur
    alpha = u / SC
    with np.errstate(divide="ignore"):
        l_k = np.where(alpha > 0, 1.0 / alpha, np.inf)
    return alpha, l_k, cur
