from pybaum.registry_entries import FUNC_DICT


def get_registry(types=None, include_defaults=True):
    """Create a pytree registry.

    Args:
        types (list): A list strings with the names of types that should be included in
            the registry, i.e. considered containers and not leaves by the functions
            that work with pytrees. Currently we support:
            - "tuple"
            - "dict"
            - "list"
            - :class:`collections.namedtuple` or :class:`typing.NamedTuple`
            - :obj:`None`
            - :class:`collections.OrderedDict`
            - "numpy.ndarray"
            - "jax.numpy.ndarray"
            - "pandas.Series"
            - "pandas.DataFrame"
        include_defaults (bool): Whether the default pytree containers "tuple", "dict"
            "list", "None", "namedtuple" and "OrderedDict" should be included even if
            not specified in `types`.

    Returns:
        dict: A pytree registry.

    """
    types = [] if types is None else types

    if include_defaults:
        default_types = {"list", "tuple", "dict", "None", "namedtuple", "OrderedDict"}
        types = list(set(types) | default_types)

    registry = {}
    for typ in types:
        new_entry = FUNC_DICT[typ]()
        registry = {**registry, **new_entry}

    return registry
