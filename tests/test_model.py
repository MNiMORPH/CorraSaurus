"""Forward-model sanity checks and a synthetic inversion-recovery test.

Run with:  python -m pytest tests/    (or just execute this file)
"""

import numpy as np

from clastattrition.model import SourceCells, predicted_counts, predicted_fractions
from clastattrition.inversion import invert


def _synthetic_cells(n_sites=40, n_liths=3, seed=0):
    """Random but reproducible source-cell distribution.

    Each site is fed by a random number of source cells of each lithology, at
    random downstream distances up to 60 km, with unit weight per cell.
    """
    rng = np.random.default_rng(seed)
    site_idx, lith_idx, distance = [], [], []
    for s in range(n_sites):
        for k in range(n_liths):
            ncells = rng.integers(5, 50)
            site_idx.append(np.full(ncells, s))
            lith_idx.append(np.full(ncells, k))
            distance.append(rng.uniform(0.0, 60_000.0, ncells))
    site_idx = np.concatenate(site_idx)
    lith_idx = np.concatenate(lith_idx)
    distance = np.concatenate(distance)
    weight = np.ones_like(distance)
    return SourceCells(site_idx, lith_idx, distance, weight, n_sites, n_liths)


def test_fractions_sum_to_one():
    cells = _synthetic_cells()
    f = predicted_fractions(np.array([10e3, 30e3, 70e3]), cells)
    assert np.allclose(f.sum(axis=1), 1.0)


def test_durable_lithology_overrepresented_downstream():
    """A larger l_k must raise a lithology's far-field share, all else equal."""
    # One site, two lithologies, identical source geometry but different l_k.
    d = np.linspace(0, 60_000, 50)
    cells = SourceCells(
        site_idx=np.zeros(2 * len(d)),
        lith_idx=np.r_[np.zeros(len(d)), np.ones(len(d))],
        distance=np.r_[d, d],
        weight=np.ones(2 * len(d)),
        n_sites=1, n_liths=2,
    )
    f = predicted_fractions(np.array([5e3, 50e3]), cells)[0]
    assert f[1] > f[0]  # durable (l=50 km) beats fragile (l=5 km)


def test_synthetic_recovery():
    """Generate data from known l_k, then recover it by inversion."""
    cells = _synthetic_cells(n_sites=60, n_liths=3, seed=1)
    l_true = np.array([8_000.0, 25_000.0, 60_000.0])
    f_obs = predicted_fractions(l_true, cells)
    counts = np.full(cells.n_sites, 100.0)

    res = invert(f_obs, cells, counts_total=counts, l0=20_000.0, objective="lsq")
    assert res.success
    rel_err = np.abs(res.l_k - l_true) / l_true
    assert np.all(rel_err < 0.05), f"recovered {res.l_k}, true {l_true}, rel_err {rel_err}"


def test_multinomial_recovery():
    cells = _synthetic_cells(n_sites=60, n_liths=3, seed=2)
    l_true = np.array([8_000.0, 25_000.0, 60_000.0])
    f_obs = predicted_fractions(l_true, cells)
    counts = np.full(cells.n_sites, 100.0)

    res = invert(f_obs, cells, counts_total=counts, objective="multinomial")
    assert res.success
    rel_err = np.abs(res.l_k - l_true) / l_true
    assert np.all(rel_err < 0.05), f"recovered {res.l_k}, true {l_true}, rel_err {rel_err}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
