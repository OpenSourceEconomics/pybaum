import pytest
from pybaum.config import IS_JAX_INSTALLED
from pybaum.registry import get_registry
from pybaum.tree_util import leaf_names
from pybaum.tree_util import tree_equal
from pybaum.tree_util import tree_flatten
from pybaum.tree_util import tree_just_flatten

if IS_JAX_INSTALLED:
    import jax.numpy as jnp
else:
    # run the tests with normal numpy instead
    import numpy as jnp


@pytest.fixture
def tree():
    return {"a": {"b": jnp.arange(4).reshape(2, 2)}, "c": jnp.ones(2)}


@pytest.fixture
def flat():
    return [0, 1, 2, 3, 1, 1]


@pytest.fixture
def registry():
    return get_registry(types=["jax.numpy.ndarray"])


def test_tree_just_flatten(tree, registry, flat):
    got = tree_just_flatten(tree, registry=registry)
    assert got == flat


def test_tree_flatten(tree, registry, flat):
    got_flat, got_treedef = tree_flatten(tree, registry=registry)
    assert got_flat == flat
    assert tree_equal(got_treedef, tree)


def test_leaf_names(tree, registry):
    got = leaf_names(tree, registry=registry)
    expected = ["a_b_0_0", "a_b_0_1", "a_b_1_0", "a_b_1_1", "c_0", "c_1"]
    assert got == expected
