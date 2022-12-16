from pybaum.config import IS_JAX_INSTALLED
from pybaum.config import IS_NUMPY_INSTALLED

if IS_JAX_INSTALLED:
    import jax.numpy as jnp

if IS_NUMPY_INSTALLED:
    import numpy as np


def get_type(obj):
    """Get type of candidate objects in a pytree.

    This function allows us to reliably identify namedtuples, NamedTuples and jax arrays
    for which standard ``type`` function does not work.

    Args:
        obj: The object to be checked

    Returns:
        type or str: The type of the object or a string with the type name.

    """
    if _is_namedtuple(obj):
        out = "namedtuple"
    elif _is_jax_array(obj):
        out = "jax.numpy.ndarray"
    else:
        out = type(obj)
    return out


def _is_namedtuple(obj):
    """Check if an object is a namedtuple.

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
    out = (
        isinstance(obj, tuple) and hasattr(obj, "_fields") and hasattr(obj, "_replace")
    )
    return out


def _is_jax_array(obj):
    """Check if an object is a jax array.

    The exact type of jax arrays has changed over time and is an implementation detail.

    Instead we rely on isinstance checks which will likely be more stable in the future.
    However, the behavior of isinstance for jax arrays has also changed over time. For
    jax versions before 0.2.21, standard numpy arrays were instances of jax arrays,
    now they are not.

    Resources:
    ----------

    - https://github.com/google/jax/issues/2115
    - https://github.com/google/jax/issues/2014
    - https://github.com/google/jax/blob/main/CHANGELOG.md#jax-0221-sept-23-2021
    - https://github.com/google/jax/blob/main/CHANGELOG.md#jax-0318-sep-26-2022

    Args:
        obj: The object to be checked

    Returns:
        bool

    """
    if not IS_JAX_INSTALLED:
        out = False
    elif IS_NUMPY_INSTALLED:
        out = isinstance(obj, jnp.ndarray) and not isinstance(obj, np.ndarray)
    else:
        out = isinstance(obj, jnp.ndarray)
    return out
