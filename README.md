# CorraSaurus

Joint inversion for **per-class Sternberg attrition (e-folding) distances** from
downstream clast fining. Given, for many sample sites, the upstream distribution
of source area vs. transport distance for each class (e.g. lithology) and the
observed downstream clast fractions and/or sizes, solve jointly for each class's
attrition length.

The method is **category-agnostic**: describe your classes with a `Categories`
(names + the integer codes they carry in your maps/tables) and pass it in — there
are no hard-coded lithologies.

## What's here

- **Modular process × data-channel inversion** (`corrasaurus.attrition.ClastInversion`):
  abrasion / fragmentation / production processes compose by adding rate
  contributions; mass / area-count / size data channels are active iff supplied.
- **Horvitz–Thompson fraction machinery** (`corrasaurus.clastdata`): turn
  area-biased (Wolman, ∝ D²) clast counts into bed number / area / mass fractions,
  with per-class clast-shape factors.
- **Lab-prior abrasion** (`abrasion_mode="lab_prior"`): a lognormal prior on each
  class's abrasion coefficient (e.g. from mill/flume experiments), with optional
  global mill→river scaling.
- **Hydraulics** (`corrasaurus.hydraulics`): Shields critical / transport shear.
- Source-distance integration model and bootstrap utilities.

## Install

```
pip install -e .          # then: import corrasaurus
pip install -e .[test]    # to run the tests
python -m pytest
```

## Status

Extracted from the Quebrada del Toro (NW Argentina) clast-attrition study via
`git filter-repo` (history preserved). The GRASS source-distance extraction lives
in its own repo (**Provenisaurus**); studies that use CorraSaurus supply their own
data + `Categories`.
