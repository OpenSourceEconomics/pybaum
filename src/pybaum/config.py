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
    import jaxlib  # noqa: F401
except ImportError:
    IS_JAXLIB_INSTALLED = False
else:
    IS_JAXLIB_INSTALLED = True
