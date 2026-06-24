"""Generic ordered classification for the attrition inversion.

The method is category-agnostic: it works with any ordered set of classes
(lithologies, provenance groups, ...).  A study builds a :class:`Categories`
from its own class list and the integer codes those classes carry in its maps
and tables (the ``lith_index`` in a geology raster / source-cell table, the
target of a field-code map, etc.).  The core never hard-codes a class list.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Categories:
    """An ordered set of classes and their external integer codes.

    Parameters
    ----------
    names : sequence of str
        Ordered class labels.  Their order defines the canonical column /
        parameter order throughout the inversion.
    indices : sequence of int
        The external integer code for each class (same order as ``names``) --
        e.g. the ``lith_index`` used in the geology raster and the source-cell
        table.  Used to map raster/table codes to positions.
    """

    names: tuple
    indices: tuple

    def __post_init__(self):
        object.__setattr__(self, "names", tuple(self.names))
        object.__setattr__(self, "indices", tuple(self.indices))
        if len(self.names) != len(self.indices):
            raise ValueError("names and indices must have the same length")
        if len(set(self.names)) != len(self.names):
            raise ValueError("category names must be unique")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("category indices must be unique")

    def __len__(self):
        return len(self.names)

    @property
    def position_by_index(self) -> dict:
        """Map external code (e.g. ``lith_index``) -> position 0..n-1."""
        return {idx: i for i, idx in enumerate(self.indices)}

    @property
    def position_by_name(self) -> dict:
        """Map class name -> position 0..n-1."""
        return {nm: i for i, nm in enumerate(self.names)}
