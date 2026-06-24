"""Nonparametric bootstrap over the raw per-clast measurements.

Each site is a finite sample of ~100 clasts, not a census.  Resampling those
clasts with replacement (optionally also resampling sites) and re-inverting many
times yields a full PDF of each lithology's attrition coefficient -- honestly
representing the one-sided / asymmetric constraints that "mean +/- SD" hides.

For count fractions this is equivalent to the multinomial sampling noise; for
mass fractions it additionally captures the dominance of the largest clasts in
Sum(D^3), which a multinomial-on-counts model cannot.
"""

from __future__ import annotations

import numpy as np

from .model import SourceCells
from .inversion import invert_alpha


def _resample_fractions(records, sites, fraction, rng, nl):
    """Resample each site's clasts with replacement -> (f_obs, n_clasts)."""
    f = np.zeros((len(sites), nl))
    ntot = np.zeros(len(sites))
    for i, s in enumerate(sites):
        lith, size = records[s]
        n = len(lith)
        ntot[i] = n
        if n == 0:
            continue
        pick = rng.integers(0, n, n)
        lp = lith[pick]
        w = np.ones(n) if fraction == "count" else size[pick] ** 3
        tot = w.sum()
        if tot > 0:
            f[i] = np.bincount(lp, weights=w, minlength=nl) / tot
    return f, ntot


def bootstrap_alpha(
    records: dict,
    cells: SourceCells,
    sites,
    n_boot: int = 1000,
    fraction: str = "mass",
    seed: int = 0,
    inverter=invert_alpha,
    **invert_kw,
) -> np.ndarray:
    """Bootstrap distribution of the attrition coefficients.

    Returns an array of shape (n_boot, n_liths) of fitted ``alpha`` [1/m].
    ``l*`` PDFs follow as ``1/alpha`` (inf where alpha == 0, i.e. durable).
    The source geometry (``cells``) is fixed; only the resampled observations
    change between draws.  ``inverter`` selects the inversion function (default
    :func:`invert_alpha`); it must return an object with an ``alpha`` attribute,
    and remaining keyword arguments are forwarded to it.
    """
    rng = np.random.default_rng(seed)
    out = np.full((n_boot, cells.n_liths), np.nan)
    for b in range(n_boot):
        f, ntot = _resample_fractions(records, sites, fraction, rng, cells.n_liths)
        res = inverter(f, cells, counts_total=ntot, **invert_kw)
        out[b] = res.alpha
    return out


def summarize(alpha_samples: np.ndarray, categories, l_max_km: float = 200.0):
    """Per-lithology percentile summary from bootstrap alpha samples.

    l* beyond ``l_max_km`` is unresolvable given the Toro's ~50 km transport
    length, so such draws are reported as 'durable' (via durable_frac) rather
    than as meaningless huge l*.  Percentiles are computed on l* censored at
    l_max_km; a percentile reported as >= l_max_km means "unresolved / durable".
    """
    rows = []
    for k, name in enumerate(categories.names):
        a = alpha_samples[:, k]
        with np.errstate(divide="ignore"):
            l_km = np.where(a > 0, 1.0 / a / 1e3, np.inf)
        durable_frac = float(np.mean(l_km > l_max_km))
        l_clip = np.minimum(l_km, l_max_km)
        pct = np.percentile(l_clip, [16, 50, 84])
        rows.append({
            "lithology": name,
            "l16_km": pct[0], "l50_km": pct[1], "l84_km": pct[2],
            "durable_frac": durable_frac, "l_max_km": l_max_km,
        })
    return rows
