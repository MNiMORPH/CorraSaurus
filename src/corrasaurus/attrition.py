"""Modular clast-attrition inversion: compose processes x data channels.

One forward model underlies everything.  A clast population leaving a source
cell evolves with downstream distance ``d`` as a *product* of per-process
factors; because each factor is ``exp(rate*d)``, processes compose by **adding**
their rate contributions.  Each process is a small function returning its
contribution to four shared rate fields:

    mass_rate   -- coarse mass  decays as exp(-mass_rate * d)
    count_rate  -- clast count  goes  as exp(-count_rate * d)  (negative => growth)
    size_rate   -- mean ln(size) = ln D0 - (size_rate/3) * <d>
    log_amp     -- log production amplitude (scales mass and count)

The data channels (mass fraction, count fraction, mean clast size) each
contribute a residual block, and are *active iff their observations are passed*.
Which processes are free and which channels are fitted are independent choices,
so this single model subsumes the former ``invert_joint`` (abrasion x
{mass,size}) and ``invert_multiprocess`` (abrasion+fragmentation+production x
{mass,count,size}).

Rates are in 1/km; ``cells.distance`` is in metres, so the forward model scales
by 1e-3 when calling the shared ``counts_and_moments`` bincount machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .model import SourceCells


def _moments(rate_perkm, cells):
    """Sum_j A_j exp(-rate*d) and Sum_j A_j d exp(-rate*d) per (site, lithology).

    Like model.counts_and_moments but works in km (rate [1/km], d [km]) and
    allows a *negative* rate -- fragmentation makes the count grow with distance
    (count_rate < 0), which the alpha=1/l>=0 guard there forbids.
    """
    d_km = cells.distance * 1e-3
    e = cells.weight * np.exp(-rate_perkm[cells.lith_idx] * d_km)
    flat = cells.site_idx * cells.n_liths + cells.lith_idx
    size = cells.n_sites * cells.n_liths
    N = np.bincount(flat, e, minlength=size).reshape(cells.n_sites, cells.n_liths)
    D = np.bincount(flat, e * d_km, minlength=size).reshape(cells.n_sites, cells.n_liths)
    return N, D


# ---------------------------------------------------------------------------
# Processes: each a small function returning its contribution to the rate fields
# ---------------------------------------------------------------------------

@dataclass
class Contribution:
    mass_rate: np.ndarray    # per lithology [1/km]
    count_rate: np.ndarray
    size_rate: np.ndarray
    log_amp: np.ndarray


def abrasion(a):
    """Surface abrasion: lose mass at rate ``a``; each clast shrinks at a/3.

    Clast *number* is conserved, but a Wolman (area-biased, ~D^2) count samples
    the shrinking clasts less and less, so the *counted* fraction declines at
    2a/3 (D^2 ~ exp(-2a/3 d)).
    """
    z = np.zeros_like(a)
    return Contribution(mass_rate=a, count_rate=2.0 * a / 3.0, size_rate=a.copy(), log_amp=z.copy())


def fragmentation(g):
    """Fragmentation: split clasts -- mass conserved, each clast shrinks at g/3.

    More clasts but each with less area, so the area-biased (Wolman) count *grows*
    at g/3 (net of the D^2 sampling), and size shrinks at g/3.
    """
    z = np.zeros_like(g)
    return Contribution(mass_rate=z.copy(), count_rate=-g / 3.0, size_rate=g.copy(), log_amp=z.copy())


def production(logc):
    """Production: relative coarse-mass yield per unit source area (an amplitude)."""
    z = np.zeros_like(logc)
    return Contribution(mass_rate=z.copy(), count_rate=z.copy(), size_rate=z.copy(), log_amp=logc)


def _sum_contributions(contribs, nl):
    total = Contribution(np.zeros(nl), np.zeros(nl), np.zeros(nl), np.zeros(nl))
    for c in contribs:
        total.mass_rate = total.mass_rate + c.mass_rate
        total.count_rate = total.count_rate + c.count_rate
        total.size_rate = total.size_rate + c.size_rate
        total.log_amp = total.log_amp + c.log_amp
    return total


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ClastResult:
    """Inert, serializable snapshot of a fit -- plain arrays and floats only.

    Holds the fitted parameters, their 1-sigma uncertainties, the predicted
    observables *at the solution*, and per-channel fit quality.  It carries no
    reference to the model (so it pickles cleanly and cannot go stale); to
    predict at *other* parameters, use ``ClastInversion.predict``.  ``theta`` is
    the raw parameter vector, which the originating model can re-expand.
    """
    # fitted parameters
    a_perkm: np.ndarray        # abrasion rate [1/km] (0 where abrasion off)
    g_perkm: np.ndarray        # fragmentation rate [1/km] (0 where off)
    c_rel: np.ndarray          # production amplitude, relative to ref (1 where off)
    D0_mm: np.ndarray          # source clast size [mm] (None if no D0 block)
    phi: float                 # abrasion mill->river scaling (None unless phi_lab)
    l_abrasion_km: np.ndarray  # 1/a [km] (inf where a==0)
    # 1-sigma uncertainties (Gauss-Newton; delta method for c, D0, l)
    a_std: np.ndarray
    g_std: np.ndarray
    c_std: np.ndarray
    D0_std: np.ndarray
    phi_std: float
    # predicted observables at the solution (None where channel inactive)
    pred_mass: np.ndarray
    pred_count: np.ndarray
    pred_size: np.ndarray
    # fit quality
    cost: float
    rmse: dict                 # per active channel: unweighted RMSE over valid sites
    success: bool
    channels: tuple
    processes: tuple
    site_mask: np.ndarray
    n_sites_used: int
    theta: np.ndarray          # raw parameter vector (re-expandable by the model)
    raw: object


# ---------------------------------------------------------------------------
# Orchestrating inversion: holds the parameter metadata; composes processes
# (free-parameter blocks) x channels (residual blocks) over the forward model.
# ---------------------------------------------------------------------------

class ClastInversion:
    """Compose processes and data channels into one joint inversion.

    Processes are switched on with the booleans ``abrasion`` / ``fragmentation``
    / ``production`` (off => that rate field stays at its identity, 0).  Data
    channels are active iff their observations are supplied (``mass_obs`` /
    ``count_obs`` / ``size_obs``).  Parameter metadata (which blocks exist, their
    bounds and initial values) is built and held on the instance.

    ``abrasion_mode`` selects how the abrasion rates are parameterised -- this is
    the single home of the mill-data coupling:

    * ``"free"`` -- independent per-lithology ``a_k``.
    * ``"phi_lab"`` -- ``a_k = phi * lab_pattern`` with one global ``phi``
      (mill->river scaling); the lab abrasion *pattern* is taken as exact.
    * ``"lab_prior"`` -- ``a_k = phi * lab_pattern * exp(delta_k)`` with a global
      ``phi`` and per-lithology log-deviations ``delta_k`` carrying a lognormal
      prior (residual ``delta_k / lab_logsigma_k``).  Tight-sigma lithologies stay
      pinned to ``phi*lab_pattern``; loose-sigma ones are free to follow the
      field data.  Reduces to ``"phi_lab"`` as ``lab_logsigma -> 0`` and
      propagates the lab spread into the uncertainties.

    ``count_likelihood`` (default ``"multinomial"``): the count channel uses the
    exact multinomial likelihood via Poisson deviance residuals (expected count
    mu_k(s)=N_s*p_k(s)); zeros contribute sqrt(2*mu) and accumulate down the
    channel, so non-detections are handled by sampling statistics rather than
    least-squares on fractions (``"lsq"`` for the latter).
    """

    def __init__(
        self,
        cells: SourceCells,
        *,
        mass_obs=None,
        count_obs=None,
        size_obs=None,
        size_count=None,
        counts_total=None,
        abrasion: bool = True,
        fragmentation: bool = False,
        production: bool = True,
        abrasion_mode: str = "free",
        lab_pattern=None,
        lab_logsigma=None,
        fit_phi: bool = True,
        ref: int = 2,
        a_bounds=(0.0, 5.0),
        g_bounds=(0.0, 5.0),
        logc_bounds=(-8.0, 8.0),
        D0_bounds_mm=(2.0, 2000.0),
        phi_bounds=(0.0, 1.0e4),
        size_weight: float = 1.0,
        count_likelihood: str = "multinomial",
        area_shape=None,
    ):
        self.cells = cells
        self.nl = nl = cells.n_liths
        self.ref = ref
        self.size_weight = size_weight
        # per-lithology a/b for the area (count) channel: a Wolman count samples
        # the projected footprint a*b = (a/b)*b^2, so a platy clast's predicted
        # area amplitude carries k_A = a/b (1 for equant; schist ~2 at 4:2:1).
        self.area_shape = (np.ones(nl) if area_shape is None
                           else np.asarray(area_shape, float))
        if count_likelihood not in ("multinomial", "lsq"):
            raise ValueError("count_likelihood must be 'multinomial' or 'lsq'")
        self.count_likelihood = count_likelihood   # count channel: exact multinomial vs least-squares

        # --- data channels (active iff observations supplied) ---
        self.mass_obs = None if mass_obs is None else np.asarray(mass_obs, float)
        self.count_obs = None if count_obs is None else np.asarray(count_obs, float)
        self.size_obs = None if size_obs is None else np.asarray(size_obs, float)
        self.size_count = None if size_count is None else np.asarray(size_count, float)
        self.channels = tuple(
            name for name, obs in
            (("mass", self.mass_obs), ("count", self.count_obs), ("size", self.size_obs))
            if obs is not None
        )
        if not self.channels:
            raise ValueError("no data channels: supply at least one of mass/count/size_obs")
        if self.size_obs is not None and self.size_count is None:
            raise ValueError("size_obs requires size_count (clasts per site,lithology)")
        self.counts_total = (np.ones(cells.n_sites) if counts_total is None
                             else np.asarray(counts_total, float))

        # --- processes ---
        if abrasion_mode not in ("free", "phi_lab", "lab_prior"):
            raise ValueError("abrasion_mode must be 'free', 'phi_lab', or 'lab_prior'")
        if abrasion_mode in ("phi_lab", "lab_prior") and lab_pattern is None:
            raise ValueError(f"abrasion_mode={abrasion_mode!r} needs lab_pattern [1/km]")
        if abrasion_mode == "lab_prior" and lab_logsigma is None:
            raise ValueError("abrasion_mode='lab_prior' needs lab_logsigma (per-lithology "
                             "log-sigma of the lab abrasion coefficient)")
        self.abrasion = abrasion
        self.fragmentation = fragmentation
        self.production = production
        self.abrasion_mode = abrasion_mode
        self.lab_pattern = None if lab_pattern is None else np.asarray(lab_pattern, float)
        self.lab_logsigma = None if lab_logsigma is None else np.asarray(lab_logsigma, float)
        self.fit_phi = fit_phi   # lab_prior: fit a global phi, or fix it at 1 (no global scaling)
        # D0 (source clast size) is needed whenever count or size is fitted.
        self.use_D0 = (self.count_obs is not None) or (self.size_obs is not None)

        self.processes = tuple(
            name for name, on in
            (("abrasion", abrasion), ("fragmentation", fragmentation), ("production", production))
            if on
        )

        # --- site mask: sources present and at least one observation row ---
        has_src = np.zeros(cells.n_sites, bool)
        has_src[np.unique(cells.site_idx)] = True
        has_obs = np.zeros(cells.n_sites, bool)
        for obs in (self.mass_obs, self.count_obs):
            if obs is not None:
                has_obs |= np.isfinite(obs).all(1) & (obs.sum(1) > 0)
        if self.size_obs is not None:
            has_obs |= (self.size_count > 0).any(1)
        self.site_mask = has_src & has_obs

        # --- precompute the production-weighted source moments (count_rate=0
        #     limit), reused when fragmentation is off ---
        self._free = [k for k in range(nl) if k != ref]

        # --- build the parameter layout (metadata held on the instance) ---
        self._blocks = []   # list of (name, size, lo, hi, x0)
        if abrasion:
            if abrasion_mode == "free":
                self._blocks.append(("a", nl, a_bounds[0], a_bounds[1], 0.1))
            elif abrasion_mode == "phi_lab":
                self._blocks.append(("phi", 1, phi_bounds[0], phi_bounds[1], 1.0))
            else:  # lab_prior: per-lithology log-deviation delta (+ optional global phi)
                if fit_phi:
                    self._blocks.append(("phi", 1, phi_bounds[0], phi_bounds[1], 1.0))
                self._blocks.append(("delta", nl, -10.0, 10.0, 0.0))
        if fragmentation:
            self._blocks.append(("g", nl, g_bounds[0], g_bounds[1], 0.1))
        if production:
            self._blocks.append(("logc", nl - 1, logc_bounds[0], logc_bounds[1], 0.0))
        if self.use_D0:
            self._blocks.append(
                ("mu0", nl, np.log(D0_bounds_mm[0]), np.log(D0_bounds_mm[1]), np.log(40.0)))
        # index slices
        self._slices = {}
        i = 0
        for name, size, *_ in self._blocks:
            self._slices[name] = slice(i, i + size)
            i += size
        self.npar = i

    # -- parameter unpacking (uses the metadata held on self) --
    def _unpack(self, theta):
        nl = self.nl
        # abrasion
        if self.abrasion:
            if self.abrasion_mode == "free":
                a = theta[self._slices["a"]]
                phi = None
            elif self.abrasion_mode == "phi_lab":
                phi = float(theta[self._slices["phi"]][0])
                a = phi * self.lab_pattern
            else:  # lab_prior: a_k = phi * lab_k * exp(delta_k); phi fixed at 1 if not fit
                phi = float(theta[self._slices["phi"]][0]) if self.fit_phi else 1.0
                delta = theta[self._slices["delta"]]
                a = phi * self.lab_pattern * np.exp(delta)
        else:
            a = np.zeros(nl); phi = None
        g = theta[self._slices["g"]] if self.fragmentation else np.zeros(nl)
        if self.production:
            logc = np.zeros(nl)
            logc[self._free] = theta[self._slices["logc"]]
        else:
            logc = np.zeros(nl)
        mu0 = theta[self._slices["mu0"]] if self.use_D0 else None
        return a, g, logc, mu0, phi

    # -- forward model: assemble process contributions, predict observables --
    def _forward(self, a, g, logc, mu0):
        contribs = []
        if self.abrasion:
            contribs.append(abrasion(a))
        if self.fragmentation:
            contribs.append(fragmentation(g))
        if self.production:
            contribs.append(production(logc))
        C = _sum_contributions(contribs, self.nl)

        # mass channel: mass = amp * sum_j A_j exp(-mass_rate d)
        Nm, _ = _moments(C.mass_rate, self.cells)
        mass = np.exp(C.log_amp)[None, :] * Nm
        massf = _safe_norm(mass)

        # count channel and size <d>: kernel exp(-count_rate d) (count_rate<0 => growth)
        Nc, Dc = _moments(C.count_rate, self.cells)
        out = {"mass": massf}
        if (self.count_obs is not None) or (self.size_obs is not None):
            with np.errstate(invalid="ignore", divide="ignore"):
                dmean_km = np.where(Nc > 0, Dc / Nc, np.nan)   # count-weighted <d> [km]
        if self.count_obs is not None:
            # Wolman area-count: amplitude c/D0 (= mass/D0^3 clasts x D0^2 area),
            # not the number-count c/D0^3; rate is in count_rate=(2a-g)/3.  The
            # footprint is a*b = k_A*D0^2, so platy clasts carry k_A = a/b.
            count = np.exp(C.log_amp[None, :] - mu0[None, :]) * self.area_shape[None, :] * Nc
            out["count"] = _safe_norm(count)
        if self.size_obs is not None:
            out["size"] = mu0[None, :] - (C.size_rate[None, :] / 3.0) * dmean_km
        return out

    def _resid(self, theta):
        a, g, logc, mu0, _ = self._unpack(theta)
        pred = self._forward(a, g, logc, mu0)
        m = self.site_mask
        parts = []
        if self.mass_obs is not None:
            w = np.sqrt(self.counts_total)[:, None]
            parts.append((w * (pred["mass"] - self.mass_obs))[m].ravel())
        if self.count_obs is not None:
            if self.count_likelihood == "multinomial":
                # exact: Poisson deviance residuals on counts (mu = N*p_pred, n = N*frac_obs)
                N = self.counts_total[:, None]
                parts.append(_poisson_dev_resid(N * self.count_obs, N * pred["count"])[m].ravel())
            else:
                w = np.sqrt(self.counts_total)[:, None]
                parts.append((w * (pred["count"] - self.count_obs))[m].ravel())
        if self.size_obs is not None:
            sd = np.sqrt(self.size_count) * (pred["size"] - self.size_obs)
            msk = (self.size_count > 0) & np.isfinite(sd)
            parts.append(self.size_weight * sd[msk])
        if self.abrasion_mode == "lab_prior":
            # lognormal prior on the abrasion shape: penalise per-lithology
            # deviation delta_k from the lab pattern by delta_k / sigma_k.
            delta = theta[self._slices["delta"]]
            parts.append(delta / self.lab_logsigma)
        return np.concatenate(parts)

    # -- public analysis: predict / residuals at arbitrary parameters --
    def predict(self, params):
        """Predicted observables ``{mass[, count][, size]}`` at ``params``.

        ``params`` may be a raw parameter vector or a :class:`ClastResult` (whose
        ``theta`` is used).  Only the active channels are returned.
        """
        theta = params.theta if isinstance(params, ClastResult) else np.asarray(params, float)
        a, g, logc, mu0, _ = self._unpack(theta)
        return self._forward(a, g, logc, mu0)

    def residuals(self, params):
        """Weighted residual vector at ``params`` (vector or :class:`ClastResult`)."""
        theta = params.theta if isinstance(params, ClastResult) else np.asarray(params, float)
        return self._resid(theta)

    def fit(self):
        x0 = np.concatenate([np.full(sz, init) for _, sz, lo, hi, init in self._blocks]) \
            if self._blocks else np.array([])
        lo = np.concatenate([np.full(sz, lo) for _, sz, lo, hi, init in self._blocks])
        hi = np.concatenate([np.full(sz, hi) for _, sz, lo, hi, init in self._blocks])
        res = least_squares(self._resid, x0, bounds=(lo, hi), method="trf")
        a, g, logc, mu0, phi = self._unpack(res.x)
        c = np.exp(logc); c = c / c[self.ref]
        with np.errstate(divide="ignore"):
            l_abr = np.where(a > 0, 1.0 / a, np.inf)

        # --- 1-sigma uncertainties from the Gauss-Newton covariance ---
        J = res.jac
        dof = max(J.shape[0] - J.shape[1], 1)
        s2 = 2.0 * res.cost / dof
        cov = np.linalg.pinv(J.T @ J) * s2
        std = np.sqrt(np.clip(np.diag(cov), 0, None))
        nl = self.nl
        a_std = np.zeros(nl); g_std = np.zeros(nl); c_std = np.zeros(nl)
        D0_std = None; phi_std = None
        if self.abrasion:
            if self.abrasion_mode == "free":
                a_std = std[self._slices["a"]]
            elif self.abrasion_mode == "phi_lab":
                phi_std = float(std[self._slices["phi"]][0])
                a_std = np.abs(self.lab_pattern) * phi_std        # delta method, a=phi*lab
            else:  # lab_prior: a_k = phi*lab_k*exp(delta_k)
                idel = self._slices["delta"].start
                if self.fit_phi:
                    phi_std = float(std[self._slices["phi"]][0])
                    iphi = self._slices["phi"].start
                    for k in range(nl):              # propagate phi & delta jointly
                        grad = np.zeros(cov.shape[0])
                        grad[iphi] = a[k] / phi       # da_k/dphi
                        grad[idel + k] = a[k]         # da_k/ddelta_k
                        a_std[k] = np.sqrt(max(float(grad @ cov @ grad), 0.0))
                else:                                 # phi fixed; a_k = lab_k*exp(delta_k)
                    a_std = a * std[self._slices["delta"]]   # delta method
        if self.fragmentation:
            g_std = std[self._slices["g"]]
        if self.production:
            c_std[self._free] = c[self._free] * std[self._slices["logc"]]   # delta method
        if self.use_D0:
            D0_std = np.exp(mu0) * std[self._slices["mu0"]]                  # delta method

        # --- predictions at the solution + per-channel unweighted RMSE ---
        pred = self._forward(a, g, logc, mu0)
        m = self.site_mask
        pred_mass = pred["mass"] if "mass" in self.channels else None
        pred_count = pred.get("count")
        pred_size = pred.get("size")
        rmse = {}
        if pred_mass is not None:
            rmse["mass"] = float(np.sqrt(np.mean((pred_mass - self.mass_obs)[m] ** 2)))
        if pred_count is not None:
            rmse["count"] = float(np.sqrt(np.mean((pred_count - self.count_obs)[m] ** 2)))
        if pred_size is not None:
            sm = (self.size_count > 0) & np.isfinite(pred_size) & np.isfinite(self.size_obs)
            rmse["size"] = float(np.sqrt(np.mean((pred_size - self.size_obs)[sm] ** 2)))

        return ClastResult(
            a_perkm=a, g_perkm=g, c_rel=c,
            D0_mm=(np.exp(mu0) if mu0 is not None else None),
            phi=phi, l_abrasion_km=l_abr,
            a_std=a_std, g_std=g_std, c_std=c_std, D0_std=D0_std, phi_std=phi_std,
            pred_mass=pred_mass, pred_count=pred_count, pred_size=pred_size,
            cost=float(res.cost), rmse=rmse, success=bool(res.success),
            channels=self.channels, processes=self.processes,
            site_mask=m.copy(), n_sites_used=int(m.sum()),
            theta=res.x.copy(), raw=res,
        )


def _safe_norm(x):
    s = x.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 0, x / s, 0.0)


def _poisson_dev_resid(n, mu):
    """Poisson deviance residual for observed count ``n``, expected count ``mu``.

    sign(n-mu)*sqrt(2*[n*ln(n/mu) - (n-mu)]); its sum of squares equals the
    deviance (-2 log-likelihood up to a constant), so least_squares on these
    residuals is an exact Poisson/multinomial fit.  A zero (n=0) gives
    sqrt(2*mu) -- the -mu (i.e. -N*p) per-site penalty that accumulates across
    the downstream sequence.  Equivalent to the multinomial for fitting the
    fractions (mu = N*p with sum(p)=1).
    """
    mu = np.maximum(mu, 1e-12)
    with np.errstate(invalid="ignore", divide="ignore"):
        term = np.where(n > 0, n * np.log(n / mu), 0.0) - (n - mu)
    return np.sign(n - mu) * np.sqrt(2.0 * np.maximum(term, 0.0))
