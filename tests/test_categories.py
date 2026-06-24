"""Tests for the generic Categories abstraction."""

import pytest

from corrasaurus.categories import Categories


def test_basic_mappings_and_order():
    c = Categories(names=("a", "b", "c"), indices=(2, 5, 9))
    assert len(c) == 3
    assert c.names == ("a", "b", "c")
    assert c.position_by_index == {2: 0, 5: 1, 9: 2}
    assert c.position_by_name == {"a": 0, "b": 1, "c": 2}


def test_coerces_lists_to_tuples():
    c = Categories(names=["x", "y"], indices=[1, 2])
    assert isinstance(c.names, tuple) and isinstance(c.indices, tuple)


def test_validation():
    with pytest.raises(ValueError):
        Categories(names=("a", "b"), indices=(1,))          # length mismatch
    with pytest.raises(ValueError):
        Categories(names=("a", "a"), indices=(1, 2))         # duplicate name
    with pytest.raises(ValueError):
        Categories(names=("a", "b"), indices=(1, 1))         # duplicate index
