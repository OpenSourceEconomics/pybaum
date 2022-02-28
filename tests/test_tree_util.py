import inspect
from collections import namedtuple
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal as aaae
from pybaum.registry import get_registry
from pybaum.tree_util import leaf_names
from pybaum.tree_util import tree_equal
from pybaum.tree_util import tree_flatten
from pybaum.tree_util import tree_map
from pybaum.tree_util import tree_multimap
from pybaum.tree_util import tree_unflatten
from pybaum.tree_util import tree_update
from pybaum.tree_util import tree_yield


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


@pytest.fixture
def extended_registry():
    types = ["pandas.DataFrame", "pandas.Series", "numpy.ndarray"]
    return get_registry(types=types)


def test_tree_flatten(example_tree, example_flat, example_treedef):
    flat, treedef = tree_flatten(example_tree)
    assert treedef == example_treedef
    _assert_list_with_arrays_is_equal(flat, example_flat)


def test_extended_tree_flatten(example_tree, extended_treedef, extended_registry):
    flat, treedef = tree_flatten(example_tree, registry=extended_registry)
    assert flat == list(range(7))
    assert tree_equal(treedef, extended_treedef)


def test_tree_flatten_with_is_leave(example_tree, extended_registry):
    flat, _ = tree_flatten(
        example_tree,
        is_leaf=lambda tree: isinstance(tree, np.ndarray),
        registry=extended_registry,
    )
    expected_flat = [0, np.array([1, 2]), 3, 4, 5, 6]
    _assert_list_with_arrays_is_equal(flat, expected_flat)


def test_tree_unflatten(example_flat, example_treedef, example_tree):
    unflat = tree_unflatten(example_treedef, example_flat)

    assert tree_equal(unflat, example_tree)


def test_extended_tree_unflatten(example_tree, extended_treedef, extended_registry):
    unflat = tree_unflatten(
        extended_treedef, list(range(7)), registry=extended_registry
    )
    assert tree_equal(unflat, example_tree)


def test_tree_unflatten_with_is_leaf(example_tree, extended_registry):
    unflat = tree_unflatten(
        example_tree,
        ([0, np.array([1, 2]), 3, 4, 5, 6]),
        is_leaf=lambda tree: isinstance(tree, np.ndarray),
        registry=extended_registry,
    )
    assert tree_equal(unflat, example_tree)


def test_tree_map():
    tree = [{"a": 1, "b": 2, "c": {"d": 3, "e": 4}}]
    calculated = tree_map(lambda x: x * 2, tree)
    expected = [{"a": 2, "b": 4, "c": {"d": 6, "e": 8}}]
    assert calculated == expected


def test_tree_multimap():
    tree = [{"a": 1, "b": 2, "c": {"d": 3, "e": 4}}]
    mapped = tree_map(lambda x: x**2, tree)
    multimapped = tree_multimap(lambda x, y: x * y, tree, tree)
    assert mapped == multimapped


def test_leaf_names(example_tree):
    names = leaf_names(example_tree, separator="*")

    expected_names = ["0*0", "0*1", "0*2*a", "0*2*b", "1"]
    assert names == expected_names


def test_extended_leaf_names(example_tree, extended_registry):
    names = leaf_names(example_tree, registry=extended_registry)
    expected_names = ["0_0", "0_1_0", "0_1_1", "0_2_a_c", "0_2_a_d", "0_2_b", "1"]
    assert names == expected_names


def test_leaf_names_with_is_leaf(example_tree, extended_registry):
    names = leaf_names(
        example_tree,
        is_leaf=lambda tree: isinstance(tree, np.ndarray),
        registry=extended_registry,
    )
    expected_names = ["0_0", "0_1", "0_2_a_c", "0_2_a_d", "0_2_b", "1"]
    assert names == expected_names


def test_iterative_flatten_and_one_step_flatten_and_unflatten(
    example_tree, extended_registry
):
    first_step_flat, first_step_treedef = tree_flatten(example_tree)
    second_step_flat, second_step_treedef = tree_flatten(
        first_step_flat, registry=extended_registry
    )
    one_step_flat, one_step_treedef = tree_flatten(
        example_tree, registry=extended_registry
    )

    assert second_step_flat == one_step_flat

    one_step_unflat = tree_unflatten(
        one_step_treedef, one_step_flat, registry=extended_registry
    )
    first_step_unflat = tree_unflatten(
        second_step_treedef, second_step_flat, registry=extended_registry
    )
    second_step_unflat = tree_unflatten(first_step_treedef, first_step_unflat)

    assert tree_equal(one_step_unflat, example_tree)
    assert tree_equal(second_step_unflat, example_tree)


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


def test_flatten_df_all_columns():
    registry = get_registry(types=["pandas.DataFrame"])
    df = pd.DataFrame(index=["a", "b", "c"])
    df["value"] = [1, 2, 3]
    df["bla"] = [4, 5, 6]

    flat, _ = tree_flatten(df, registry=registry)

    assert flat == [1, 4, 2, 5, 3, 6]


def test_tree_yield(example_tree, example_treedef, example_flat):
    generator, treedef = tree_yield(example_tree)

    assert treedef == example_treedef
    assert inspect.isgenerator(generator)
    for a, b in zip(generator, example_flat):
        if isinstance(a, (np.ndarray, pd.Series)):
            aaae(a, b)
        else:
            assert a == b


def test_flatten_with_none():
    flat, treedef = tree_flatten(None)
    assert flat == []
    assert treedef is None


def test_leaf_names_with_none():
    names = leaf_names(None)
    assert names == []


def test_flatten_with_namedtuple():
    bla = namedtuple("bla", ["a", "b"])(1, 2)
    flat, _ = tree_flatten(bla)
    assert flat == [1, 2]


def test_names_with_namedtuple():
    bla = namedtuple("bla", ["a", "b"])(1, 2)
    names = leaf_names(bla)
    assert names == ["a", "b"]


def test_flatten_with_ordered_dict():
    d = OrderedDict({"a": 1, "b": 2})
    flat, _ = tree_flatten(d)
    assert flat == [1, 2]


def test_names_with_ordered_dict():
    d = OrderedDict({"a": 1, "b": 2})
    names = leaf_names(d)
    assert names == ["a", "b"]
