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


def test_lab_prior_reduces_to_phi_lab():
    """As lab_logsigma -> 0 the lognormal-prior mode collapses to point phi_lab."""
    cells = _synthetic_cells(n_sites=70, seed=20)
    nl = cells.n_liths
    lab = np.array([0.006, 0.36, 0.002, 0.006, 0.018])
    D0 = np.full(nl, 40.0)
    obs = _generate(cells, 4.0 * lab, np.zeros(nl), np.ones(nl), D0, channels=("mass", "size"))
    kw = dict(abrasion=True, fragmentation=False, production=False, lab_pattern=lab, **obs)
    rp = AT.ClastInversion(cells, abrasion_mode="phi_lab", **kw).fit()
    rt = AT.ClastInversion(cells, abrasion_mode="lab_prior",
                           lab_logsigma=np.full(nl, 1e-3), **kw).fit()
    assert np.isclose(rt.phi, rp.phi, rtol=1e-2), (rt.phi, rp.phi)
    assert np.allclose(rt.a_perkm, rp.a_perkm, rtol=1e-2)


def test_lab_prior_fit_phi_false():
    """fit_phi=False drops the global scaling: a_k = lab_k * exp(delta_k), no phi."""
    cells = _synthetic_cells(n_sites=80, seed=22)
    nl = cells.n_liths
    lab = np.array([0.006, 0.36, 0.002, 0.006, 0.018])
    D0 = np.full(nl, 40.0)
    a = lab * np.exp(np.array([0.2, -0.3, 0.1, -0.2, 0.15]))
    obs = _generate(cells, a, np.zeros(nl), np.ones(nl), D0, channels=("mass", "size"))
    inv = AT.ClastInversion(cells, abrasion_mode="lab_prior", lab_pattern=lab,
                            lab_logsigma=np.full(nl, 5.0), fit_phi=False,
                            abrasion=True, fragmentation=False, production=False, **obs)
    assert "phi" not in inv._slices                 # no global-phi parameter block
    r = inv.fit()
    assert r.success
    # with a loose prior the data dominates; the resolvable lithology recovers
    # (a tighter prior biases it toward lab_pattern -- expected for a penalty term)
    assert np.isclose(r.a_perkm[1], a[1], rtol=0.05), (r.a_perkm[1], a[1])


def test_lab_prior_recovers_rates():
    """With a loose prior the data dominates and the well-resolved lithologies
    recover known a_k = phi*lab*exp(delta).  Durable (tiny-a) lithologies are
    only weakly constrained by mass+size -- the real physics, not a bug -- so the
    recovery assertion is on the resolvable subset."""
    cells = _synthetic_cells(n_sites=90, seed=21)
    nl = cells.n_liths
    lab = np.array([0.006, 0.36, 0.002, 0.006, 0.018])
    D0 = np.array([45., 35., 46., 36., 28.])
    a = 5.0 * lab * np.exp(np.array([0.3, -0.4, 0.2, -0.5, 0.1]))
    obs = _generate(cells, a, np.zeros(nl), np.ones(nl), D0, channels=("mass", "size"))
    r = AT.ClastInversion(cells, abrasion_mode="lab_prior", lab_pattern=lab,
                          lab_logsigma=np.full(nl, 10.0),     # loose => data dominates
                          abrasion=True, fragmentation=False, production=False, **obs).fit()
    assert r.success
    resolvable = a > 0.05
    assert np.allclose(r.a_perkm[resolvable], a[resolvable], rtol=0.03), (r.a_perkm, a)


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


def test_poisson_deviance_residual():
    """Deviance residual: 0 when n==mu; -sqrt(2*mu) for a zero count."""
    from clastattrition.attrition import _poisson_dev_resid
    r = _poisson_dev_resid(np.array([5.0, 0.0, 0.0, 3.0]), np.array([5.0, 2.0, 4.5, 3.0]))
    assert np.isclose(r[0], 0.0)
    assert np.isclose(r[1], -2.0)        # -sqrt(2*2)
    assert np.isclose(r[2], -3.0)        # -sqrt(2*4.5)
    assert np.isclose(r[3], 0.0)


def test_count_likelihood_modes_recover():
    """Both the exact multinomial and lsq count likelihoods recover known rates."""
    cells = _synthetic_cells(n_sites=80, seed=30)
    nl = cells.n_liths
    a = np.array([0.005, 0.30, 0.002, 0.010, 0.05])
    g = np.array([0.02, 0.10, 0.005, 0.30, 0.08])
    c = np.array([1.0, 0.5, 1.0, 3.0, 0.4])
    D0 = np.array([45., 35., 46., 36., 28.])
    obs = _generate(cells, a, g, c, D0)
    assert _raises_value_error(lambda: AT.ClastInversion(cells, count_likelihood="bad", **obs))
    for lik in ("multinomial", "lsq"):
        r = AT.ClastInversion(cells, abrasion=True, fragmentation=True, production=True,
                              count_likelihood=lik, **obs).fit()
        assert r.success
        assert np.allclose(r.a_perkm, a, atol=1e-3) and np.allclose(r.g_perkm, g, atol=1e-3), \
            (lik, r.a_perkm, r.g_perkm)


def test_area_shape_scales_predicted_count():
    """area_shape (k_A=a/b) multiplies a lithology's predicted area amplitude."""
    cells = _synthetic_cells(seed=3)
    nl = cells.n_liths
    shp = (cells.n_sites, nl)
    a = np.full(nl, 0.05); g = np.zeros(nl); c = np.ones(nl); D0 = np.full(nl, 40.0)
    kw = dict(count_obs=np.zeros(shp), size_obs=np.full(shp, np.nan),
              size_count=np.ones(shp), abrasion=True, production=True)
    base = AT.ClastInversion(cells, **kw)
    f = np.ones(nl); f[2] = 2.0                      # treat lithology 2 as platy
    shaped = AT.ClastInversion(cells, area_shape=f, **kw)
    pb = base._forward(a, g, np.log(c), np.log(D0))["count"]
    ps = shaped._forward(a, g, np.log(c), np.log(D0))["count"]
    # normalization cancels in a ratio, so lith 2 / lith 0 scales by exactly k_A
    rb = pb[:, 2] / pb[:, 0]; rs = ps[:, 2] / ps[:, 0]
    good = np.isfinite(rb) & np.isfinite(rs) & (rb > 0)
    assert good.any()
    assert np.allclose(rs[good], 2.0 * rb[good])


def _raises_value_error(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


def test_input_validation():
    """The constructor rejects incoherent configurations."""
    cells = _synthetic_cells(seed=10)
    z = np.zeros((cells.n_sites, cells.n_liths))
    assert _raises_value_error(lambda: AT.ClastInversion(cells))                       # no channel
    assert _raises_value_error(lambda: AT.ClastInversion(cells, size_obs=z))           # size needs count
    assert _raises_value_error(lambda: AT.ClastInversion(cells, mass_obs=z, abrasion_mode="bad"))
    assert _raises_value_error(lambda: AT.ClastInversion(cells, mass_obs=z, abrasion_mode="phi_lab"))  # no pattern
    lab = np.ones(cells.n_liths)
    assert _raises_value_error(  # lab_prior needs lab_logsigma
        lambda: AT.ClastInversion(cells, mass_obs=z, abrasion_mode="lab_prior", lab_pattern=lab))


def test_fragmentation_inflates_count_not_mass():
    """The fingerprint that lets count separate g: fragmentation raises a
    lithology's count share but leaves its mass share untouched."""
    # one site, two lithologies with identical source geometry; fragment only lith 0.
    d = np.linspace(500.0, 14_000.0, 40)
    cells = SourceCells(
        site_idx=np.zeros(2 * len(d)),
        lith_idx=np.r_[np.zeros(len(d)), np.ones(len(d))],
        distance=np.r_[d, d], weight=np.ones(2 * len(d)),
        n_sites=1, n_liths=2,
    )
    inv = AT.ClastInversion(cells, mass_obs=np.zeros((1, 2)), count_obs=np.zeros((1, 2)),
                            abrasion=False, fragmentation=True, production=False)
    pred = inv._forward(np.zeros(2), np.array([0.3, 0.0]), np.zeros(2), np.log(np.full(2, 40.0)))
    assert np.allclose(pred["mass"][0], 0.5)        # mass share unchanged: still 50/50
    assert pred["count"][0, 0] > 0.5                 # fragmenting lithology over-counted


def test_uncertainty_responds_to_noise():
    """Sigma is ~0 for a noise-free exact fit and grows once noise is added."""
    cells = _synthetic_cells(n_sites=80, seed=12)
    nl = cells.n_liths
    obs = _generate(cells, np.full(nl, 0.05), np.zeros(nl), np.ones(nl),
                    np.full(nl, 40.0), channels=("mass", "size"))
    kw = dict(abrasion=True, fragmentation=False, production=False)
    r0 = AT.ClastInversion(cells, **kw, **obs).fit()
    rng = np.random.default_rng(0)
    noisy = dict(obs)
    m = obs["mass_obs"] + rng.normal(0, 0.02, obs["mass_obs"].shape)
    noisy["mass_obs"] = np.clip(m, 1e-6, None)
    noisy["mass_obs"] /= noisy["mass_obs"].sum(1, keepdims=True)
    rN = AT.ClastInversion(cells, **kw, **noisy).fit()
    assert r0.a_std.max() < 1e-3
    assert rN.a_std.max() > r0.a_std.max()
    assert np.all(np.isfinite(rN.a_std)) and np.all(rN.a_std >= 0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
