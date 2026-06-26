"""Load the derived observations and GIS-extracted source-cell distributions.

Two inputs feed the inversion:

1. **Observations** -- ``clast_observations.csv`` (from
   ``clastdata.build_observations``): per-site Horvitz-Thompson number/area/mass
   fractions and the per-site clast total used to weight the inversion.

2. **Source cells** -- a long-format CSV produced by the GIS extraction with
   columns ``site, lith_index, distance_m, weight``: one row per source raster
   cell upstream of a sample site, giving its class, downstream flow distance,
   and production weight.

``load_dataset`` aligns the two on the site name, in a single shared order.
These loaders are category-agnostic: pass a :class:`~.categories.Categories`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .model import SourceCells


@dataclass
class Dataset:
    sites: np.ndarray            # (n_sites,) site names, the shared order
    f_obs: np.ndarray            # (n_sites, n_liths) observed fractions
    counts_total: np.ndarray     # (n_sites,) clasts counted per site
    coords: np.ndarray           # (n_sites, 2) lon, lat
    cells: SourceCells           # upstream source-cell distribution
    lith_names: tuple            # category names, in canonical order


def load_source_cells(path: str, site_order, categories) -> SourceCells:
    """Load the long-format source-cell CSV and align it to ``site_order``.

    Rows whose ``site`` is not in ``site_order`` (e.g. unmatched names) or whose
    ``lith_index`` is not a modelled source class are dropped with no error.

    The four-column schema is the same whether each row is one source raster
    cell or a pre-binned ``(site, lith, distance-bin)`` histogram (the latter is
    what Provenisaurus now emits by default): a histogram row just carries a
    summed ``weight`` and the bin's weight-mean ``distance_m``.  Granularity is
    transparent here because the forward model only ever reads ``weight`` and
    ``distance``, never a cell count, so both inputs flow through this one path
    unchanged.  :func:`~.model.reduce_cells` remains available either way --
    idempotent on already-binned input, and the standard coarsener for raw
    per-cell tables from any other source.
    """
    pbi = categories.position_by_index
    df = pd.read_csv(path)
    site_to_row = {name: i for i, name in enumerate(site_order)}
    keep = df["site"].isin(site_to_row) & df["lith_index"].isin(pbi)
    df = df[keep]
    site_idx = df["site"].map(site_to_row).to_numpy()
    lith_idx = df["lith_index"].map(pbi).to_numpy()
    distance = df["distance_m"].to_numpy(dtype=float)
    weight = (
        df["weight"].to_numpy(dtype=float)
        if "weight" in df.columns
        else np.ones(len(df))
    )
    return SourceCells(
        site_idx, lith_idx, distance, weight,
        n_sites=len(site_order), n_liths=len(categories),
    )


def load_dataset(observations_path: str, source_path: str, categories,
                 fraction: str = "mass") -> Dataset:
    """Load and align clast observations and source-cell distributions.

    Restricts to the sites present in *both* the observations (clast counts)
    and the GIS source-cell table (i.e. the in-watershed sites).

    Parameters
    ----------
    observations_path : str
        clast_observations.csv from scripts/build_observations.py.
    source_path : str
        source_cells.csv from the GIS extraction.
    categories : Categories
        the ordered class set (names + lith_index codes).
    fraction : {"mass", "area", "number"}
        Which observed (Horvitz-Thompson-corrected) bed fractions to invert
        against.  "mass" is the Sternberg-correct observable; "area" is the raw
        Wolman tally (it estimates the bed area fraction, and matches the AGU
        2020 analysis); "number" is the by-number fraction.  Legacy "count" is
        accepted as an alias for "area".
    """
    from .clastdata import fractions_matrix

    obs = pd.read_csv(observations_path).set_index("site")
    src_sites = set(pd.read_csv(source_path, usecols=["site"])["site"].unique())
    sites = [s for s in obs.index if s in src_sites]
    obs = obs.loc[sites]

    f_obs = fractions_matrix(obs, categories, fraction)
    counts_total = obs["n_clasts"].to_numpy(dtype=float)
    cells = load_source_cells(source_path, sites, categories)
    return Dataset(np.array(sites), f_obs, counts_total, None, cells, categories.names)
