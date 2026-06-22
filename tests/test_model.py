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


def test_reduce_cells_is_lossless():
    """Histogram reduction must reproduce fractions to ~machine precision."""
    from clastattrition.model import reduce_cells
    cells = _synthetic_cells(n_sites=15, n_liths=4, seed=11)
    red = reduce_cells(cells, bin_width_m=12.0)
    assert red.n_cells < cells.n_cells
    for l in ([5e3, 15e3, 40e3, 80e3], [2e3, 200e3, 1e3, 50e3]):
        f_full = predicted_fractions(np.array(l), cells)
        f_red = predicted_fractions(np.array(l), red)
        assert np.allclose(f_full, f_red, atol=1e-6), np.abs(f_full - f_red).max()


def test_analytic_jacobian_matches_finite_difference():
    """The analytic Jacobian of the alpha residuals must match finite diffs."""
    from clastattrition.inversion import _resid_jac, _prepare
    cells = _synthetic_cells(n_sites=20, n_liths=3, seed=5)
    l_true = np.array([8_000.0, 25_000.0, 60_000.0])
    f_obs = predicted_fractions(l_true, cells)
    counts = np.full(cells.n_sites, 100.0)
    f_obs, counts, mask = _prepare(f_obs, counts, cells)

    a = np.array([1 / 12_000.0, 1 / 30_000.0, 1 / 80_000.0])
    r0, J = _resid_jac(a, f_obs, counts, cells, mask)
    Jfd = np.zeros_like(J)
    for m in range(len(a)):
        da = a[m] * 1e-6 + 1e-12
        ap = a.copy(); ap[m] += da
        rp, _ = _resid_jac(ap, f_obs, counts, cells, mask)
        Jfd[:, m] = (rp - r0) / da
    assert np.allclose(J, Jfd, rtol=1e-4, atol=1e-9), np.abs(J - Jfd).max()


def test_alpha_inversion_recovers_truth():
    from clastattrition.inversion import invert_alpha, invert_alpha_coorddescent
    cells = _synthetic_cells(n_sites=60, n_liths=3, seed=7)
    l_true = np.array([8_000.0, 25_000.0, 60_000.0])
    f_obs = predicted_fractions(l_true, cells)
    counts = np.full(cells.n_sites, 100.0)
    res = invert_alpha(f_obs, cells, counts_total=counts)
    assert res.success
    rel = np.abs(res.l_k - l_true) / l_true
    assert np.all(rel < 0.05), (res.l_k, l_true)
    # coordinate descent should agree (compare in alpha; the most durable
    # parameter is only marginally constrained, so allow a small atol).
    alpha_cd, _, _ = invert_alpha_coorddescent(f_obs, cells, counts_total=counts)
    assert np.allclose(alpha_cd, res.alpha, rtol=0.05, atol=2e-6), (alpha_cd, res.alpha)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
