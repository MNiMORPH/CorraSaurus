"""Tests for the modular process x channel inversion (clastattrition.attrition).

All synthetic and self-contained (no external data), matching test_model.py.

Run with:  python -m pytest tests/test_attrition.py   (or execute this file)
"""

import copy
import pickle

import numpy as np

from clastattrition.model import SourceCells
from clastattrition import attrition as AT


def _synthetic_cells(n_sites=60, n_liths=5, seed=0):
    """Reproducible random source-cell distribution (distances up to 15 km)."""
    rng = np.random.default_rng(seed)
    site_idx, lith_idx, distance = [], [], []
    for s in range(n_sites):
        for k in range(n_liths):
            ncells = rng.integers(5, 40)
            site_idx.append(np.full(ncells, s))
            lith_idx.append(np.full(ncells, k))
            distance.append(rng.uniform(0.0, 15_000.0, ncells))
    si = np.concatenate(site_idx); li = np.concatenate(lith_idx)
    dd = np.concatenate(distance); w = np.ones_like(dd)
    return SourceCells(si, li, dd, w, n_sites, n_liths)


def _generate(cells, a, g, c, D0, *, channels=("mass", "count", "size")):
    """Truth observables from the forward model, for the requested channels."""
    nl = cells.n_liths
    shp = (cells.n_sites, nl)
    gen = AT.ClastInversion(
        cells,
        mass_obs=np.zeros(shp), count_obs=np.zeros(shp),
        size_obs=np.full(shp, np.nan), size_count=np.ones(shp),
        abrasion=True, fragmentation=True, production=True,
    )
    pred = gen._forward(a, g, np.log(c), np.log(D0))
    obs = {}
    if "mass" in channels:
        obs["mass_obs"] = pred["mass"]
    if "count" in channels:
        obs["count_obs"] = pred["count"]
    if "size" in channels:
        obs["size_obs"] = pred["size"]
        sc = np.ones(shp); sc[~gen.site_mask] = 0
        obs["size_count"] = sc
    return obs


# --- physics ---------------------------------------------------------------

def test_mass_conserved_under_fragmentation():
    """Pure fragmentation must not change any lithology's mass fraction."""
    cells = _synthetic_cells(seed=1)
    nl = cells.n_liths
    inv = AT.ClastInversion(cells, mass_obs=np.zeros((cells.n_sites, nl)),
                            abrasion=False, fragmentation=True, production=False)
    base = inv._forward(np.zeros(nl), np.zeros(nl), np.zeros(nl), None)["mass"]
    frag = inv._forward(np.zeros(nl), np.full(nl, 0.3), np.zeros(nl), None)["mass"]
    assert np.allclose(np.nan_to_num(base), np.nan_to_num(frag))


def test_synthetic_recovery_all_processes():
    """Recover known a, g, c, D0 from noise-free mass+count+size observations."""
    cells = _synthetic_cells(n_sites=80, seed=2)
    a = np.array([0.005, 0.30, 0.002, 0.010, 0.05])
    g = np.array([0.02, 0.10, 0.005, 0.30, 0.08])
    c = np.array([1.0, 0.5, 1.0, 3.0, 0.4])
    D0 = np.array([45., 35., 46., 36., 28.])
    obs = _generate(cells, a, g, c, D0)
    r = AT.ClastInversion(cells, abrasion=True, fragmentation=True, production=True,
                          **obs).fit()
    assert r.success
    assert np.allclose(r.a_perkm, a, atol=1e-4), (r.a_perkm, a)
    assert np.allclose(r.g_perkm, g, atol=1e-4), (r.g_perkm, g)
    assert np.allclose(r.c_rel, c, rtol=1e-3), (r.c_rel, c)
    assert np.allclose(r.D0_mm, D0, rtol=1e-3), (r.D0_mm, D0)


def test_phi_lab_recovers_scaling():
    """abrasion_mode='phi_lab' recovers the global mill->river scaling phi."""
    cells = _synthetic_cells(n_sites=80, seed=3)
    lab = np.array([0.006, 0.36, 0.002, 0.006, 0.018])
    phi_true = 7.0
    a = phi_true * lab
    D0 = np.array([45., 35., 46., 36., 28.])
    obs = _generate(cells, a, np.zeros(cells.n_liths), np.ones(cells.n_liths), D0,
                    channels=("mass", "size"))
    r = AT.ClastInversion(cells, abrasion=True, fragmentation=False, production=False,
                          abrasion_mode="phi_lab", lab_pattern=lab, **obs).fit()
    assert r.success
    assert np.isclose(r.phi, phi_true, rtol=1e-3), (r.phi, phi_true)


def test_reproduces_invert_joint():
    """abrasion x {mass,size} must match the standalone invert_joint."""
    from clastattrition.inversion import invert_joint
    cells = _synthetic_cells(n_sites=70, seed=4)
    a = np.array([0.0025, 1.2, 0.07, 0.17, 0.8])      # within invert_joint's l bounds
    D0 = np.array([45., 35., 46., 36., 28.])
    obs = _generate(cells, a, np.zeros(cells.n_liths), np.ones(cells.n_liths), D0,
                    channels=("mass", "size"))
    counts = np.full(cells.n_sites, 100.0)
    rj = invert_joint(cells, f_obs=obs["mass_obs"], counts_total=counts,
                      size_mean=obs["size_obs"], size_count=obs["size_count"],
                      mode="joint", n=0.0)
    r = AT.ClastInversion(cells, counts_total=counts,
                          mass_obs=obs["mass_obs"], size_obs=obs["size_obs"],
                          size_count=obs["size_count"],
                          abrasion=True, fragmentation=False, production=False,
                          a_bounds=(1e3 / 400e3, 200.0)).fit()
    assert np.allclose(r.l_abrasion_km, rj.l_k / 1e3, rtol=1e-3), (r.l_abrasion_km, rj.l_k / 1e3)


# --- modularity ------------------------------------------------------------

def test_inactive_process_is_identity():
    """Switching a process off pins its rate at the identity (0 / c=1)."""
    cells = _synthetic_cells(seed=5)
    obs = _generate(cells, np.full(cells.n_liths, 0.05), np.zeros(cells.n_liths),
                    np.ones(cells.n_liths), np.full(cells.n_liths, 40.0),
                    channels=("mass", "size"))
    r = AT.ClastInversion(cells, abrasion=True, fragmentation=False, production=False,
                          **obs).fit()
    assert np.all(r.g_perkm == 0.0)          # fragmentation off
    assert np.allclose(r.c_rel, 1.0)         # production off
    assert "fragmentation" not in r.processes and "production" not in r.processes


def test_channels_follow_supplied_observations():
    """A channel is active iff its observations are passed."""
    cells = _synthetic_cells(seed=6)
    nl = cells.n_liths
    obs = _generate(cells, np.full(nl, 0.05), np.zeros(nl), np.ones(nl), np.full(nl, 40.0))
    mass_only = AT.ClastInversion(cells, mass_obs=obs["mass_obs"], production=False)
    assert mass_only.channels == ("mass",)
    both = AT.ClastInversion(cells, mass_obs=obs["mass_obs"], size_obs=obs["size_obs"],
                             size_count=obs["size_count"], production=False)
    assert both.channels == ("mass", "size")
    # parameter count grows with the size channel (adds the D0 block)
    assert both.npar > mass_only.npar


# --- result object ---------------------------------------------------------

def test_result_is_inert_and_picklable():
    """ClastResult holds no model reference and pickles as plain data."""
    cells = _synthetic_cells(seed=7)
    nl = cells.n_liths
    obs = _generate(cells, np.full(nl, 0.05), np.full(nl, 0.05), np.ones(nl),
                    np.full(nl, 40.0))
    r = AT.ClastInversion(cells, abrasion=True, fragmentation=True, production=True,
                          **obs).fit()
    # no attribute anywhere on the result references the model
    assert not any(isinstance(v, AT.ClastInversion) for v in vars(r).values())
    snap = copy.copy(r); snap.raw = None          # raw is the scipy result; drop for purity
    r2 = pickle.loads(pickle.dumps(snap))
    assert np.allclose(r2.a_perkm, r.a_perkm)


def test_enrichment_fields_present():
    """Uncertainties, per-channel RMSE, and fitted predictions are populated."""
    cells = _synthetic_cells(seed=8)
    nl = cells.n_liths
    obs = _generate(cells, np.full(nl, 0.05), np.full(nl, 0.05), np.ones(nl),
                    np.full(nl, 40.0))
    r = AT.ClastInversion(cells, abrasion=True, fragmentation=True, production=True,
                          **obs).fit()
    assert r.a_std.shape == (nl,) and r.g_std.shape == (nl,)
    assert r.D0_std is not None and np.all(np.isfinite(r.a_std))
    assert set(r.rmse) == set(r.channels)
    assert r.pred_mass is not None and r.pred_count is not None and r.pred_size is not None


def test_predict_and_residuals_match_fit():
    """The model's predict/residuals at the solution reproduce the stored fit."""
    cells = _synthetic_cells(seed=9)
    nl = cells.n_liths
    obs = _generate(cells, np.full(nl, 0.05), np.full(nl, 0.05), np.ones(nl),
                    np.full(nl, 40.0))
    inv = AT.ClastInversion(cells, abrasion=True, fragmentation=True, production=True, **obs)
    r = inv.fit()
    assert np.allclose(inv.predict(r)["mass"], r.pred_mass)
    assert np.allclose(inv.residuals(r), r.raw.fun)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
