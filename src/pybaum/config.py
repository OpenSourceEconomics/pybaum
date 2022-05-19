try:
    import numpy as np  # noqa: F401
except ImportError:
    IS_NUMPY_INSTALLED = False
else:
    IS_NUMPY_INSTALLED = True


try:
    import pandas as pd  # noqa: F401
except ImportError:
    IS_PANDAS_INSTALLED = False
else:
    IS_PANDAS_INSTALLED = True


try:
    import jax  # noqa: F401
    import jaxlib  # noqa: F401
except ImportError:
    IS_JAX_INSTALLED = False
else:
    IS_JAX_INSTALLED = True
