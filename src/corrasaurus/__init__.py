"""CorraSaurus -- per-class Sternberg attrition-distance inversion from clast fining.

Solve for per-class (e.g. per-lithology) Sternberg attrition (e-folding)
distances from downstream clast fractions across many sample catchments at once.
The method is category-agnostic: supply a :class:`Categories` describing your
classes (names + the integer codes they carry in your maps/tables).
"""

from .categories import Categories
from .model import SourceCells, predicted_counts, predicted_fractions, reduce_cells
from .inversion import invert, InversionResult, invert_alpha
from .attrition import ClastInversion, ClastResult
from .clastdata import (
    load_code_map, build_observations, load_clast_records, size_moments,
    fractions_matrix,
)
from .io import Dataset, load_source_cells, load_dataset

__all__ = [
    "Categories",
    "SourceCells", "predicted_counts", "predicted_fractions", "reduce_cells",
    "invert", "InversionResult", "invert_alpha",
    "ClastInversion", "ClastResult",
    "load_code_map", "build_observations", "load_clast_records", "size_moments",
    "fractions_matrix",
    "Dataset", "load_source_cells", "load_dataset",
]
