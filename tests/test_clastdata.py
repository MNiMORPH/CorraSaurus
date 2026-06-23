"""Tests for the Horvitz-Thompson number/area/mass fractions in clastdata."""

import numpy as np
import pandas as pd
import pytest

from clastattrition.clastdata import build_observations, fractions_matrix
from clastattrition.lithology import NAMES, POSITION

openpyxl = pytest.importorskip("openpyxl")

# Two lithologies, hand-chosen clasts, so the three fractions are all distinct.
# granite (G): sizes 10, 10 mm ; quartzite (M): size 40 mm
# code map: "G" -> granite (lith_index 2), "M" -> quartzite (lith_index 4)
_SIZES = {"G": [10.0, 10.0], "M": [40.0]}


def _make_workbook(path):
    """Write a minimal one-sheet workbook matching the real layout.

    Per-clast rows start at row 9; col B = size (mm), col C = field code.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "S1"
    row = 9
    for code, sizes in _SIZES.items():
        for s in sizes:
            ws.cell(row=row, column=2, value=s)    # col B
            ws.cell(row=row, column=3, value=code)  # col C
            row += 1
    wb.save(path)


def test_horvitz_thompson_fractions(tmp_path):
    xlsx = tmp_path / "counts.xlsx"
    _make_workbook(str(xlsx))
    code_map = {"G": 2, "M": 4}  # granite, quartzite

    df = build_observations(str(xlsx), code_map)
    assert list(df.index) == ["S1"]
    g, m = POSITION["granite"], POSITION["quartzite"]

    # raw tally: 2 granite + 1 quartzite
    assert int(df.loc["S1", "n_clasts"]) == 3

    # area fraction = raw tally fraction
    assert df.loc["S1", f"area_frac_{NAMES[g]}"] == pytest.approx(2 / 3)
    assert df.loc["S1", f"area_frac_{NAMES[m]}"] == pytest.approx(1 / 3)

    # number fraction: weight 1/D^2  -> G: 2/100, M: 1/1600
    wg, wm = 2 / 100.0, 1 / 1600.0
    assert df.loc["S1", f"number_frac_{NAMES[g]}"] == pytest.approx(wg / (wg + wm))
    assert df.loc["S1", f"number_frac_{NAMES[m]}"] == pytest.approx(wm / (wg + wm))

    # mass fraction: weight D  -> G: 10+10=20, M: 40
    assert df.loc["S1", f"mass_frac_{NAMES[g]}"] == pytest.approx(20 / 60)
    assert df.loc["S1", f"mass_frac_{NAMES[m]}"] == pytest.approx(40 / 60)

    # the three fractions for granite are genuinely distinct (no equal-D collapse)
    fr = [df.loc["S1", f"{k}_frac_{NAMES[g]}"] for k in ("number", "area", "mass")]
    assert len(set(np.round(fr, 6))) == 3


def test_mass_is_sumD_not_sumD3(tmp_path):
    """Mass fraction must use Sum D (D^3 * 1/D^2), not the biased Sum D^3."""
    xlsx = tmp_path / "counts.xlsx"
    _make_workbook(str(xlsx))
    df = build_observations(str(xlsx), {"G": 2, "M": 4})
    g = POSITION["granite"]
    sumD = 20 / 60                      # correct Horvitz-Thompson mass fraction
    sumD3 = (2 * 10 ** 3) / (2 * 10 ** 3 + 40 ** 3)  # the old, biased value
    assert df.loc["S1", f"mass_frac_{NAMES[g]}"] == pytest.approx(sumD)
    assert not np.isclose(sumD, sumD3)  # they really differ


def test_fractions_matrix_kinds_and_alias():
    cols = {}
    for kind in ("number", "area", "mass"):
        for nm in NAMES:
            cols[f"{kind}_frac_{nm}"] = [0.2]
    df = pd.DataFrame(cols, index=["S1"])

    area = fractions_matrix(df, "area")
    assert area.shape == (1, len(NAMES))
    # legacy "count" aliases to "area"
    np.testing.assert_array_equal(fractions_matrix(df, "count"), area)

    with pytest.raises(ValueError):
        fractions_matrix(df, "volume")
