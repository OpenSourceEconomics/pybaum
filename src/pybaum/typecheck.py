from collections import namedtuple


def get_type(obj):
    """namdetuple aware type check.

    As in JAX we treat collections.namedtuple and typing.NamedTuple both as
    namedtuple but the exact type is preserved in the unflatten function.

    namedtuples are discovered by being instances of tuple and having a
    ``_fields`` attribute as suggested by Raymond Hettinger
    `here <https://bugs.python.org/issue7796>`_.

    Moreover we check for the presence of a ``_replace`` method because we need when
    unflattening pytrees.

    This can produce false positives but in most cases would still result in desired
    behavior.

    Args:
        obj: The object to be checked

    Returns:
        bool

    """
    if isinstance(obj, tuple) and hasattr(obj, "_fields") and hasattr(obj, "_replace"):
        out = namedtuple
    else:
        out = type(obj)
    return out
