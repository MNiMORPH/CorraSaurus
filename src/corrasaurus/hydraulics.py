"""Bed-mobility hydraulics: critical Shields stress and representative transport
shear stress per site.

A representative transport shear stress is taken as ``transport_ratio`` (default
1.2) times the critical, threshold-of-motion shear stress for the site's median
grain size D50 (Shields):

    tau_c = tau*_c * (rho_s - rho_w) * g * D            [Pa]
    tau_transport = transport_ratio * tau_c

This is the flow energy at which a site's gravel actually moves (and abrades), so
it sets where on the abrasion-vs-shear-stress curve the field sits -- the bridge
between the lab flume rates (measured at fixed shear stress) and the field.

Defaults: tau*_c = 0.0497 (Wong & Parker 2006, paired with the median grain D50),
rho_s = 2650 kg/m^3 (crystalline / quartz), rho_w = 1000, g = 9.81.  A
slope-dependent tau*_c (e.g. Lamb et al. 2008) would raise it on steep reaches;
not included here.  D50 (not D84) is used as the characteristic grain, for
consistency with the near-uniform-sediment calibration of tau*_c.
"""

from __future__ import annotations

import numpy as np


def critical_shear_stress(D_m, shields_crit=0.0497, rho_s=2650.0, rho_w=1000.0, g=9.81):
    """Threshold-of-motion bed shear stress [Pa] for grain diameter ``D_m`` [m]."""
    return shields_crit * (rho_s - rho_w) * g * np.asarray(D_m, dtype=float)


def transport_shear_stress(D50_m, transport_ratio=1.2, **kwargs):
    """Representative transport shear stress [Pa] = ``transport_ratio`` * tau_c(D50)."""
    return transport_ratio * critical_shear_stress(D50_m, **kwargs)


def site_d50_mm(records, sites):
    """Median clast diameter [mm] per site, from the per-clast size records.

    ``records`` is ``{site: (lith_idx, size_mm)}`` from
    :func:`corrasaurus.clastdata.load_clast_records`.
    """
    d50 = np.full(len(sites), np.nan)
    for i, s in enumerate(sites):
        sizes = records[s][1]
        if len(sizes):
            d50[i] = float(np.median(sizes))
    return d50


def site_transport_shear(records, sites, transport_ratio=1.2, shields_crit=0.0497,
                         rho_s=2650.0, rho_w=1000.0, g=9.81):
    """Per-site D50, critical shear stress, and representative transport shear.

    Returns ``(d50_mm, tau_c_Pa, tau_transport_Pa)``, each an array over ``sites``.
    """
    d50_mm = site_d50_mm(records, sites)
    tau_c = critical_shear_stress(d50_mm / 1e3, shields_crit, rho_s, rho_w, g)
    tau_t = transport_ratio * tau_c
    return d50_mm, tau_c, tau_t
