"""Implement functionality similar to jax.tree_util in pure Python.

The functions are not completely identical to jax. The most notable differences are:

- Instead of a global registry of pytree nodes, most functions have a registry argument.
- The treedef containing information to unflatten pytrees is implemented differently.

"""
from pybaum.equality import EQUALITY_CHECKERS
from pybaum.registry import get_registry
from pybaum.typecheck import get_type


def tree_flatten(tree, is_leaf=None, registry=None):
    """Flatten a pytree and create a treedef.

    Args:
        tree: a pytree to flatten.
        is_leaf (callable or None): An optionally specified function that will be called
            at each flattening step. It should return a boolean, which indicates whether
            the flattening should traverse the current object, or if it should be
            stopped immediately, with the whole subtree being treated as a leaf.
        registry (dict or None): A pytree container registry that determines
            which types are considered container objects that should be flattened.
            ``is_leaf`` can override this in the sense that types that are in the
            registry are still considered a leaf but it cannot declare something a
            container that is not in the registry. None means that the default registry
            is used, i.e. that dicts, tuples and lists are considered containers.
            "extended" means that in addition numpy arrays and params DataFrames are
            considered containers. Passing a dictionary where the keys are types and the
            values are dicts with the entries "flatten", "unflatten" and "names" allows
            to completely override the default registries.

    Returns:
        A pair where the first element is a list of leaf values and the second
        element is a treedef representing the structure of the flattened tree.

    """
    registry = _process_pytree_registry(registry)
    is_leaf = _process_is_leaf(is_leaf)

    flat = _tree_flatten(tree, is_leaf=is_leaf, registry=registry)
    # unflatten the flat tree to make a copy
    treedef = tree_unflatten(tree, flat, is_leaf=is_leaf, registry=registry)
    return flat, treedef


def tree_just_flatten(tree, is_leaf=None, registry=None):
    """Flatten a pytree without creating a treedef.

    Args:
        tree: a pytree to flatten.
        is_leaf (callable or None): An optionally specified function that will be called
            at each flattening step. It should return a boolean, which indicates whether
            the flattening should traverse the current object, or if it should be
            stopped immediately, with the whole subtree being treated as a leaf.
        registry (dict or None): A pytree container registry that determines
            which types are considered container objects that should be flattened.
            ``is_leaf`` can override this in the sense that types that are in the
            registry are still considered a leaf but it cannot declare something a
            container that is not in the registry. None means that the default registry
            is used, i.e. that dicts, tuples and lists are considered containers.
            "extended" means that in addition numpy arrays and params DataFrames are
            considered containers. Passing a dictionary where the keys are types and the
            values are dicts with the entries "flatten", "unflatten" and "names" allows
            to completely override the default registries.

    Returns:
        A pair where the first element is a list of leaf values and the second
        element is a treedef representing the structure of the flattened tree.

    """
    registry = _process_pytree_registry(registry)
    is_leaf = _process_is_leaf(is_leaf)

    flat = _tree_flatten(tree, is_leaf=is_leaf, registry=registry)
    return flat


def _tree_flatten(tree, is_leaf, registry):
    out = []
    tree_type = get_type(tree)

    if tree_type not in registry or is_leaf(tree):
        out.append(tree)
    else:
        subtrees, _ = registry[tree_type]["flatten"](tree)
        for subtree in subtrees:
            if get_type(subtree) in registry:
                out += _tree_flatten(subtree, is_leaf, registry)
            else:
                out.append(subtree)
    return out


def tree_yield(tree, is_leaf=None, registry=None):
    """Yield leafs from a pytree and create the tree definition.

    Args:
        tree: a pytree.
        is_leaf (callable or None): An optionally specified function that will be called
            at each yield step. It should return a boolean, which indicates whether
            the generator should traverse the current object, or if it should be
            stopped immediately, with the whole subtree being treated as a leaf.
        registry (dict or None): A pytree container registry that determines
            which types are considered container objects that should be yielded.
            ``is_leaf`` can override this in the sense that types that are in the
            registry are still considered a leaf but it cannot declare something a
            container that is not in the registry. None means that the default registry
            is used, i.e. that dicts, tuples and lists are considered containers.
            "extended" means that in addition numpy arrays and params DataFrames are
            considered containers. Passing a dictionary where the keys are types and the
            values are dicts with the entries "flatten", "unflatten" and "names" allows
            to completely override the default registries.

    Returns:
        A pair where the first element is a generator of leaf values and the second
        element is a treedef representing the structure of the flattened tree.

    """
    registry = _process_pytree_registry(registry)
    is_leaf = _process_is_leaf(is_leaf)

    flat = _tree_yield(tree, is_leaf=is_leaf, registry=registry)
    return flat, tree


def tree_just_yield(tree, is_leaf=None, registry=None):
    """Yield leafs from a pytree without creating a treedef.

    Args:
        tree: a pytree.
        is_leaf (callable or None): An optionally specified function that will be called
            at each yield step. It should return a boolean, which indicates whether
            the generator should traverse the current object, or if it should be
            stopped immediately, with the whole subtree being treated as a leaf.
        registry (dict or None): A pytree container registry that determines
            which types are considered container objects that should be yielded.
            ``is_leaf`` can override this in the sense that types that are in the
            registry are still considered a leaf but it cannot declare something a
            container that is not in the registry. None means that the default registry
            is used, i.e. that dicts, tuples and lists are considered containers.
            "extended" means that in addition numpy arrays and params DataFrames are
            considered containers. Passing a dictionary where the keys are types and the
            values are dicts with the entries "flatten", "unflatten" and "names" allows
            to completely override the default registries.

    Returns:
        A generator of leaf values.

    """
    registry = _process_pytree_registry(registry)
    is_leaf = _process_is_leaf(is_leaf)

    flat = _tree_yield(tree, is_leaf=is_leaf, registry=registry)
    return flat


def _tree_yield(tree, is_leaf, registry):
    out = []
    tree_type = get_type(tree)

    if tree_type not in registry or is_leaf(tree):
        yield tree
    else:
        subtrees, _ = registry[tree_type]["flatten"](tree)
        for subtree in subtrees:
            if get_type(subtree) in registry:
                yield from _tree_yield(subtree, is_leaf, registry)
            else:
                yield subtree
    return out


def tree_unflatten(treedef, leaves, is_leaf=None, registry=None):
    """Reconstruct a pytree from the treedef and a list of leaves.

    The inverse of :func:`tree_flatten`.

    Args:
        treedef: the treedef to with information needed for reconstruction.
        leaves (list): the list of leaves to use for reconstruction. The list must match
            the leaves of the treedef.
        is_leaf (callable or None): An optionally specified function that will be called
            at each flattening step. It should return a boolean, which indicates whether
            the flattening should traverse the current object, or if it should be
            stopped immediately, with the whole subtree being treated as a leaf.
        registry (dict or None): A pytree container registry that determines
            which types are considered container objects that should be flattened.
            `is_leaf` can override this in the sense that types that are in the
            registry are still considered a leaf but it cannot declare something a
            container that is not in the registry. None means that the default registry
            is used, i.e. that dicts, tuples and lists are considered containers.
            "extended" means that in addition numpy arrays and params DataFrames are
            considered containers. Passing a dictionary where the keys are types and the
            values are dicts with the entries "flatten", "unflatten" and "names" allows
            to completely override the default registries.

    Returns:
        The reconstructed pytree, containing the ``leaves`` placed in the structure
        described by ``treedef``.

    """
    registry = _process_pytree_registry(registry)
    is_leaf = _process_is_leaf(is_leaf)
    return _tree_unflatten(treedef, leaves, is_leaf=is_leaf, registry=registry)


def _tree_unflatten(treedef, leaves, is_leaf, registry):
    leaves = iter(leaves)
    tree_type = get_type(treedef)

    if tree_type not in registry or is_leaf(treedef):
        return next(leaves)
    else:
        items, info = registry[tree_type]["flatten"](treedef)
        unflattened_items = []
        for item in items:
            if get_type(item) in registry:
                unflattened_items.append(
                    _tree_unflatten(item, leaves, is_leaf=is_leaf, registry=registry)
                )
            else:
                unflattened_items.append(next(leaves))
        return registry[tree_type]["unflatten"](info, unflattened_items)


def tree_map(func, tree, is_leaf=None, registry=None):
    """Apply func to all leaves in tree.

    Args:
        func (callable): Function applied to each leaf in the tree.
        tree: A pytree.
        is_leaf (callable or None): An optionally specified function that will be called
            at each flattening step. It should return a boolean, which indicates whether
            the flattening should traverse the current object, or if it should be
            stopped immediately, with the whole subtree being treated as a leaf.
        registry (dict or None): A pytree container registry that determines
            which types are considered container objects that should be flattened.
            `is_leaf` can override this in the sense that types that are in the
            registry are still considered a leaf but it cannot declare something a
            container that is not in the registry. None means that the default registry
            is used, i.e. that dicts, tuples and lists are considered containers.
            "extended" means that in addition numpy arrays and params DataFrames are
            considered containers. Passing a dictionary where the keys are types and the
            values are dicts with the entries "flatten", "unflatten" and "names" allows
            to completely override the default registries.
    Returns:
        modified copy of tree.

    """
    flat, treedef = tree_flatten(tree, is_leaf=is_leaf, registry=registry)
    modified = [func(i) for i in flat]
    new_tree = tree_unflatten(treedef, modified, is_leaf=is_leaf, registry=registry)
    return new_tree


def tree_multimap(func, *trees, is_leaf=None, registry=None):
    """Apply func to leaves of multiple pytrees.

    Args:
        func (callable): Function applied to each leaf corresponding leaves of
            multiple py trees.
        trees: An arbitrary number of pytrees. All trees need to have the same
            structure.
        is_leaf (callable or None): An optionally specified function that will be called
            at each flattening step. It should return a boolean, which indicates whether
            the flattening should traverse the current object, or if it should be
            stopped immediately, with the whole subtree being treated as a leaf.
        registry (dict or None): A pytree container registry that determines
            which types are considered container objects that should be flattened.
            `is_leaf` can override this in the sense that types that are in the
            registry are still considered a leaf but it cannot declare something a
            container that is not in the registry. None means that the default registry
            is used, i.e. that dicts, tuples and lists are considered containers.
            "extended" means that in addition numpy arrays and params DataFrames are
            considered containers. Passing a dictionary where the keys are types and the
            values are dicts with the entries "flatten", "unflatten" and "names" allows
            to completely override the default registries.
    Returns:
        tree with the same structure as the elements in trees.

    """
    flat_trees, treedefs = [], []
    for tree in trees:
        flat, treedef = tree_flatten(tree, is_leaf=is_leaf, registry=registry)
        flat_trees.append(flat)
        treedefs.append(treedef)

    for treedef in treedefs:
        if treedef != treedefs[0]:
            raise ValueError("All trees must have the same structure.")

    modified = [func(*item) for item in zip(*flat_trees)]

    new_trees = tree_unflatten(
        treedefs[0], modified, is_leaf=is_leaf, registry=registry
    )
    return new_trees


def leaf_names(tree, is_leaf=None, registry=None, separator="_"):
    """Construct names for leaves in a pytree.

    Args:
        tree: a pytree to flatten.
        is_leaf (callable or None): An optionally specified function that will be called
            at each flattening step. It should return a boolean, which indicates whether
            the flattening should traverse the current object, or if it should be
            stopped immediately, with the whole subtree being treated as a leaf.
        registry (dict or None): A pytree container registry that determines
            which types are considered container objects that should be flattened.
            `is_leaf` can override this in the sense that types that are in the
            registry are still considered a leaf but it cannot declare something a
            container that is not in the registry. None means that the default registry
            is used, i.e. that dicts, tuples and lists are considered containers.
            "extended" means that in addition numpy arrays and params DataFrames are
            considered containers. Passing a dictionary where the keys are types and the
            values are dicts with the entries "flatten", "unflatten" and "names" allows
            to completely override the default registries.
        separator (str): String that separates the building blocks of the leaf name.
    Returns:
        list: List of strings with names for pytree leaves.

    """
    registry = _process_pytree_registry(registry)
    is_leaf = _process_is_leaf(is_leaf)
    leaf_names = _leaf_names(
        tree, is_leaf=is_leaf, registry=registry, separator=separator
    )
    return leaf_names


def _leaf_names(tree, is_leaf, registry, separator, prefix=None):
    out = []
    tree_type = get_type(tree)

    if tree_type not in registry or is_leaf(tree):
        out.append(prefix)
    else:
        subtrees, info = registry[tree_type]["flatten"](tree)
        names = registry[tree_type]["names"](tree)
        for name, subtree in zip(names, subtrees):
            if get_type(subtree) in registry:
                out += _leaf_names(
                    subtree,
                    is_leaf=is_leaf,
                    registry=registry,
                    separator=separator,
                    prefix=_add_prefix(prefix, name, separator),
                )
            else:
                out.append(_add_prefix(prefix, name, separator))
    return out


def _add_prefix(prefix, string, separator):
    if prefix not in (None, ""):
        out = separator.join([prefix, string])
    else:
        out = string
    return out


def _process_pytree_registry(registry):
    registry = registry if registry is not None else get_registry()
    return registry


def _process_is_leaf(is_leaf):
    if is_leaf is None:
        return lambda tree: False  # noqa: U100
    else:
        return is_leaf


def tree_equal(tree, other, is_leaf=None, registry=None, equality_checkers=None):
    """Determine if two pytrees are equal.

    Two pytrees are considered equal if their leaves are equal and the names of their
    leaves are equal. While this definition of equality might not always make sense
    it makes sense in most cases and can be implemented relatively easily.

    Args:
        tree: A pytree.
        other: Another pytree.
        is_leaf (callable or None): An optionally specified function that will be called
            at each flattening step. It should return a boolean, which indicates whether
            the flattening should traverse the current object, or if it should be
            stopped immediately, with the whole subtree being treated as a leaf.
        registry (dict or None): A pytree container registry that determines
            which types are considered container objects that should be flattened.
            `is_leaf` can override this in the sense that types that are in the
            registry are still considered a leaf but it cannot declare something a
            container that is not in the registry. None means that the default registry
            is used, i.e. that dicts, tuples and lists are considered containers.
            "extended" means that in addition numpy arrays and params DataFrames are
            considered containers. Passing a dictionary where the keys are types and the
            values are dicts with the entries "flatten", "unflatten" and "names" allows
            to completely override the default registries.
        equality_checkers (dict, None): A dictionary where keys are types and values are
            functions which assess equality for the type of object.

    Returns:
        bool

    """
    equality_checkers = (
        EQUALITY_CHECKERS
        if equality_checkers is None
        else {**EQUALITY_CHECKERS, **equality_checkers}
    )

    first_flat = tree_just_flatten(tree, is_leaf=is_leaf, registry=registry)
    second_flat = tree_just_flatten(other, is_leaf=is_leaf, registry=registry)

    first_names = leaf_names(tree, is_leaf=is_leaf, registry=registry)
    second_names = leaf_names(tree, is_leaf=is_leaf, registry=registry)

    equal = first_names == second_names

    if equal:
        for first, second in zip(first_flat, second_flat):
            check_func = equality_checkers.get(get_type(first), lambda a, b: a == b)
            equal = equal and check_func(first, second)
            if not equal:
                break

    return equal


def tree_update(tree, other, is_leaf=None, registry=None):
    """Update leaves in a pytree with leaves from another pytree.

    The second pytree must be compatible with the first one but can be smaller. For
    example, lists can be shorter, dictionaries can contain subsets of entries, etc.

    Args:
        tree: A pytree.
        other: Another pytree.
        is_leaf (callable or None): An optionally specified function that will be called
            at each flattening step. It should return a boolean, which indicates whether
            the flattening should traverse the current object, or if it should be
            stopped immediately, with the whole subtree being treated as a leaf.
        registry (dict or None): A pytree container registry that determines
            which types are considered container objects that should be flattened.
            `is_leaf` can override this in the sense that types that are in the
            registry are still considered a leaf but it cannot declare something a
            container that is not in the registry. None means that the default registry
            is used, i.e. that dicts, tuples and lists are considered containers.
            "extended" means that in addition numpy arrays and params DataFrames are
            considered containers. Passing a dictionary where the keys are types and the
            values are dicts with the entries "flatten", "unflatten" and "names" allows
            to completely override the default registries.
    Returns:
        Updated pytree.

    """
    first_flat, first_treedef = tree_flatten(tree, is_leaf=is_leaf, registry=registry)
    first_names = leaf_names(tree, is_leaf=is_leaf, registry=registry)
    first_dict = dict(zip(first_names, first_flat))

    other_flat, _ = tree_flatten(other, is_leaf=is_leaf, registry=registry)
    other_names = leaf_names(other, is_leaf=is_leaf, registry=registry)
    other_dict = dict(zip(other_names, other_flat))

    combined = list({**first_dict, **other_dict}.values())

    out = tree_unflatten(first_treedef, combined, is_leaf=is_leaf, registry=registry)
    return out
