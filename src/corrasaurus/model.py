"""Forward model: predicted clast-count fractions from Sternberg attrition.

Physical model
--------------
A clast loses mass exponentially with downstream transport distance
(Sternberg's law).  We invert clast-count *fractions* by lithology, so we treat
the per-lithology e-folding distance ``l_k`` as an **effective count-abundance
attrition distance**: the distance over which a lithology's contribution to the
counted (gravel-sized) population decays by a factor of ``e``.  A durable
lithology has a large ``l_k`` and stays over-represented far from its source.

For sample site ``s`` the predicted *count* of lithology ``k`` is a sum over all
upstream source cells ``j`` of that lithology:

    N_k(s) = sum_j  A_j * exp(-d_js / l_k)

where ``A_j`` is the per-cell source-production weight (binary mapped-source
area in v1; a continuous "clast-generation potential" later) and ``d_js`` is the
downstream flow distance from the source cell to the sample site.  The predicted
fraction is the normalised version,

    f_k(s) = N_k(s) / sum_k' N_k'(s).

Data representation
-------------------
Source cells are held in flat ("long") arrays so the whole problem -- every
source cell of every lithology feeding every site -- is a handful of NumPy
vectors and the forward model is fully vectorised:

    site_idx[c]  : row index of the sample site fed by source cell c
    lith_idx[c]  : column index (canonical lithology order) of source cell c
    distance[c]  : downstream flow distance d_js  [metres]
    weight[c]    : source-production weight A_j

See ``corrasaurus.lithology`` for the canonical lithology ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SourceCells:
    """Flat (long-format) description of every source cell feeding every site.

    All four arrays share length ``n_cells``.  ``n_sites`` and ``n_liths`` give
    the shape of the predicted-fraction matrix the forward model returns.
    """

    site_idx: np.ndarray   # int, in [0, n_sites)
    lith_idx: np.ndarray   # int, in [0, n_liths)
    distance: np.ndarray   # float, metres
    weight: np.ndarray     # float, source-production weight
    n_sites: int
    n_liths: int

    def __post_init__(self) -> None:
        n = len(self.site_idx)
        for name in ("lith_idx", "distance", "weight"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} length {len(getattr(self, name))} != {n}")
        self.site_idx = np.asarray(self.site_idx, dtype=np.intp)
        self.lith_idx = np.asarray(self.lith_idx, dtype=np.intp)
        self.distance = np.asarray(self.distance, dtype=float)
        self.weight = np.asarray(self.weight, dtype=float)
        # Precompute the flattened (site, lith) bin index for fast accumulation.
        self._flat_idx = self.site_idx * self.n_liths + self.lith_idx
        self._flat_size = self.n_sites * self.n_liths

    @property
    def n_cells(self) -> int:
        return len(self.site_idx)


def predicted_counts(l_k: np.ndarray, cells: SourceCells) -> np.ndarray:
    """Predicted (un-normalised) clast counts ``N_k(s)``.

    Parameters
    ----------
    l_k : array, shape (n_liths,)
        Effective attrition e-folding distance per lithology [metres], in
        canonical lithology order.  Must be strictly positive.
    cells : SourceCells

    Returns
    -------
    N : array, shape (n_sites, n_liths)
    """
    l_k = np.asarray(l_k, dtype=float)
    if np.any(l_k <= 0):
        raise ValueError("attrition distances l_k must be strictly positive")
    contrib = cells.weight * np.exp(-cells.distance / l_k[cells.lith_idx])
    flat = np.bincount(cells._flat_idx, weights=contrib, minlength=cells._flat_size)
    return flat.reshape(cells.n_sites, cells.n_liths)


def reduce_cells(cells: SourceCells, bin_width_m: float = 12.0) -> SourceCells:
    """Losslessly collapse source cells into per-(site, lith) distance bins.

    Cells at (nearly) the same distance contribute identical exp(-alpha d), so
    summing their weights first is algebraically exact.  Each bin keeps the
    summed weight and the weight-*mean* distance, so the leading term is exact
    and the residual error is O((alpha * bin_width)^2) -- negligible at bin
    widths <= the DEM cell size.  Every site/lithology stays fully represented;
    this is data aggregation, not downsampling.

    Reduces the forward pass from millions of cells to ~10^4-10^5 bins, which is
    what makes bootstrap and MCMC tractable.
    """
    dbin = np.floor(cells.distance / bin_width_m).astype(np.intp)
    nb = int(dbin.max()) + 1
    key = (cells.site_idx * cells.n_liths + cells.lith_idx) * nb + dbin
    size = cells.n_sites * cells.n_liths * nb
    W = np.bincount(key, weights=cells.weight, minlength=size)
    WD = np.bincount(key, weights=cells.weight * cells.distance, minlength=size)
    nz = np.nonzero(W)[0]
    w = W[nz]
    d = WD[nz] / w                      # weight-mean distance within each bin
    sl = nz // nb
    lith = sl % cells.n_liths
    site = sl // cells.n_liths
    return SourceCells(site, lith, d, w, cells.n_sites, cells.n_liths)


def counts_and_moments(alpha: np.ndarray, cells: SourceCells):
    """Predicted counts and their first distance-moment, in the alpha = 1/l_k
    parameterisation.

    Returns
    -------
    N : array (n_sites, n_liths)
        N_k(s) = sum_j A_j exp(-alpha_k d_js)  -- the Laplace transform of the
        source-distance distribution at alpha_k.
    D : array (n_sites, n_liths)
        D_k(s) = sum_j A_j d_js exp(-alpha_k d_js) = -dN_k/dalpha_k  -- the first
        moment, used to build the analytic Jacobian.

    Both are computed with two bincounts over all source cells at once.
    """
    alpha = np.asarray(alpha, dtype=float)
    if np.any(alpha < 0):
        raise ValueError("attrition coefficients alpha must be non-negative")
    e = cells.weight * np.exp(-alpha[cells.lith_idx] * cells.distance)
    shape = (cells.n_sites, cells.n_liths)
    N = np.bincount(cells._flat_idx, weights=e, minlength=cells._flat_size).reshape(shape)
    D = np.bincount(cells._flat_idx, weights=e * cells.distance,
                    minlength=cells._flat_size).reshape(shape)
    return N, D


def predicted_fractions(l_k: np.ndarray, cells: SourceCells) -> np.ndarray:
    """Predicted clast-count fractions ``f_k(s)``, rows summing to 1.

    Sites with no upstream source of any modelled lithology yield a row of
    zeros (no prediction); callers should mask these against the observations.
    """
    N = predicted_counts(l_k, cells)
    totals = N.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(totals > 0, N / totals, 0.0)
    return f
