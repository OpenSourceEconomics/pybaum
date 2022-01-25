import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from pybaum.tree_util import leaf_names
from pybaum.tree_util import tree_equal
from pybaum.tree_util import tree_flatten
from pybaum.tree_util import tree_map
from pybaum.tree_util import tree_multimap
from pybaum.tree_util import tree_unflatten
from pybaum.tree_util import tree_update


@pytest.fixture
def example_tree():
    return (
        [0, np.array([1, 2]), {"a": pd.Series([3, 4], index=["c", "d"]), "b": 5}],
        6,
    )


@pytest.fixture
def example_flat():
    return [0, np.array([1, 2]), pd.Series([3, 4], index=["c", "d"]), 5, 6]


@pytest.fixture
def example_treedef():
    return (["*", "*", {"a": "*", "b": "*"}], "*")


@pytest.fixture
def extended_treedef():
    return (
        [
            "*",
            np.array(["*", "*"]),
            {"a": pd.Series(["*", "*"], index=["c", "d"]), "b": "*"},
        ],
        "*",
    )


def test_tree_flatten(example_tree, example_flat, example_treedef):
    flat, treedef = tree_flatten(example_tree)
    assert treedef == example_treedef
    _assert_list_with_arrays_is_equal(flat, example_flat)


def test_tree_unflatten(example_flat, example_treedef, example_tree):
    unflat = tree_unflatten(example_treedef, example_flat)

    assert tree_equal(unflat, example_tree)


def test_tree_map():
    tree = [{"a": 1, "b": 2, "c": {"d": 3, "e": 4}}]
    calculated = tree_map(lambda x: x * 2, tree)
    expected = [{"a": 2, "b": 4, "c": {"d": 6, "e": 8}}]
    assert calculated == expected


def test_tree_multimap():
    tree = [{"a": 1, "b": 2, "c": {"d": 3, "e": 4}}]
    mapped = tree_map(lambda x: x ** 2, tree)
    multimapped = tree_multimap(lambda x, y: x * y, tree, tree)
    assert mapped == multimapped


def test_leaf_names(example_tree):
    names = leaf_names(example_tree, separator="*")

    expected_names = ["0*0", "0*1", "0*2*a", "0*2*b", "1"]
    assert names == expected_names


def test_tree_update(example_tree):
    other = ([7, np.array([8, 9]), {"b": 10}], 11)
    updated = tree_update(example_tree, other)

    expected = (
        [7, np.array([8, 9]), {"a": pd.Series([3, 4], index=["c", "d"]), "b": 10}],
        11,
    )
    assert tree_equal(updated, expected)


def _assert_list_with_arrays_is_equal(list1, list2):
    for first, second in zip(list1, list2):
        if isinstance(first, np.ndarray):
            aaae(first, second)
        elif isinstance(first, (pd.DataFrame, pd.Series)):
            assert first.equals(second)
        else:
            assert first == second
