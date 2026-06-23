"""Tests for the Shields / transport-shear-stress hydraulics."""

import numpy as np

from clastattrition.hydraulics import (
    critical_shear_stress, transport_shear_stress, site_d50_mm, site_transport_shear)


def test_critical_shear_stress_known_value():
    # tau*_c=0.0497 (Wong & Parker 2006) * (2650-1000) * 9.81 * 0.05 m = 40.22 Pa
    assert np.isclose(critical_shear_stress(0.05), 0.0497 * 1650 * 9.81 * 0.05)
    assert np.isclose(critical_shear_stress(0.05), 40.22, atol=0.05)


def test_transport_is_1p2_times_critical():
    assert np.isclose(transport_shear_stress(0.05), 1.2 * critical_shear_stress(0.05))


def test_site_d50_is_median():
    records = {"A": (np.array([0, 1, 0]), np.array([10.0, 20.0, 30.0])),
               "B": (np.array([2]), np.array([50.0])),
               "C": (np.array([], int), np.array([]))}
    d50 = site_d50_mm(records, ["A", "B", "C"])
    assert d50[0] == 20.0 and d50[1] == 50.0 and np.isnan(d50[2])


def test_site_transport_shear_shapes_and_scaling():
    records = {"A": (np.array([0]), np.array([64.0]))}   # cobble D50 = 64 mm
    d50, tau_c, tau_t = site_transport_shear(records, ["A"])
    assert d50[0] == 64.0
    assert np.isclose(tau_c[0], 0.0497 * 1650 * 9.81 * 0.064)
    assert np.isclose(tau_t[0], 1.2 * tau_c[0])
