"""Functions to check equality of pytree leaves."""
from pybaum.config import IS_JAX_INSTALLED
from pybaum.config import IS_NUMPY_INSTALLED
from pybaum.config import IS_PANDAS_INSTALLED


if IS_NUMPY_INSTALLED:
    import numpy as np


if IS_PANDAS_INSTALLED:
    import pandas as pd

if IS_JAX_INSTALLED:
    import jaxlib


EQUALITY_CHECKERS = {}


if IS_NUMPY_INSTALLED:
    EQUALITY_CHECKERS[np.ndarray] = lambda a, b: bool((a == b).all())


if IS_PANDAS_INSTALLED:
    EQUALITY_CHECKERS[pd.Series] = lambda a, b: a.equals(b)
    EQUALITY_CHECKERS[pd.DataFrame] = lambda a, b: a.equals(b)


if IS_JAX_INSTALLED:
    EQUALITY_CHECKERS[jaxlib.xla_extension.DeviceArray] = lambda a, b: bool(
        (a == b).all()
    )
