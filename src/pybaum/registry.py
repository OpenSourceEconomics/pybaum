from pybaum.registry_entries import FUNC_DICT


def get_registry(types=None, options=None, include_defaults=True):
    """Create a pytree registry.

    Args:
        types (list): A list strings with the names of types that should be included in
            the registry, i.e. considered containers and not leaves by the functions
            that work with pytrees. Currently we support:
            - "tuple"
            - "dict"
            - "list"
            - "numpy.ndarray"
            - "pandas.Series"
            - "pandas.DataFrame"
        options (dict): Option dictionary where the keys are names of types and the
            values are keyword arguments that influence how containers are flattened
            and unflattened.
        include_defaults (bool): Whether the default pytree containers "tuple", "dict"
            and "list" should be included even if not specified in `types`.

    Returns:
        dict: A pytree registry.

    """
    types = [] if types is None else types

    if include_defaults:
        types = list(set(types) | {"list", "tuple", "dict"})

    options = {} if options is None else options

    registry = {}
    for typ in types:
        new_entry = FUNC_DICT[typ](**options.get(typ, {}))
        registry = {**registry, **new_entry}

    return registry
