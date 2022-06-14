import itertools
from collections import namedtuple
from collections import OrderedDict
from itertools import product

from pybaum.config import IS_JAX_INSTALLED
from pybaum.config import IS_NUMPY_INSTALLED
from pybaum.config import IS_PANDAS_INSTALLED

if IS_NUMPY_INSTALLED:
    import numpy as np

if IS_PANDAS_INSTALLED:
    import pandas as pd

if IS_JAX_INSTALLED:
    import jax
    import jaxlib


def _none():
    """Create registry entry for NoneType."""
    entry = {
        type(None): {
            "flatten": lambda tree: ([], None),  # noqa: U100
            "unflatten": lambda aux_data, children: None,  # noqa: U100
            "names": lambda tree: [],  # noqa: U100
        }
    }
    return entry


def _list():
    """Create registry entry for list."""
    entry = {
        list: {
            "flatten": lambda tree: (tree, None),
            "unflatten": lambda aux_data, children: children,  # noqa: U100
            "names": lambda tree: [f"{i}" for i in range(len(tree))],
        },
    }
    return entry


def _dict():
    """Create registry entry for dict."""
    entry = {
        dict: {
            "flatten": lambda tree: (list(tree.values()), list(tree)),
            "unflatten": lambda aux_data, children: dict(zip(aux_data, children)),
            "names": lambda tree: list(map(str, list(tree))),
        },
    }
    return entry


def _tuple():
    """Create registry entry for tuple."""
    entry = {
        tuple: {
            "flatten": lambda tree: (list(tree), None),
            "unflatten": lambda aux_data, children: tuple(children),  # noqa: U100
            "names": lambda tree: [f"{i}" for i in range(len(tree))],
        },
    }
    return entry


def _namedtuple():
    """Create registry entry for namedtuple and NamedTuple."""
    entry = {
        namedtuple: {
            "flatten": lambda tree: (list(tree), tree),
            "unflatten": _unflatten_namedtuple,
            "names": lambda tree: list(tree._fields),
        },
    }
    return entry


def _unflatten_namedtuple(aux_data, leaves):
    replacements = dict(zip(aux_data._fields, leaves))
    out = aux_data._replace(**replacements)
    return out


def _ordereddict():
    """Create registry entry for OrderedDict."""
    entry = {
        OrderedDict: {
            "flatten": lambda tree: (list(tree.values()), list(tree)),
            "unflatten": lambda aux_data, children: OrderedDict(
                zip(aux_data, children)
            ),
            "names": lambda tree: list(map(str, list(tree))),
        },
    }
    return entry


def _numpy_array():
    """Create registry entry for numpy.ndarray."""

    if IS_NUMPY_INSTALLED:
        entry = {
            np.ndarray: {
                "flatten": lambda arr: (arr.flatten().tolist(), arr.shape),
                "unflatten": lambda aux_data, leaves: np.array(leaves).reshape(
                    aux_data
                ),
                "names": _array_element_names,
            },
        }
    else:
        entry = {}
    return entry


def _array_element_names(arr):
    dim_names = [map(str, range(n)) for n in arr.shape]
    names = list(map("_".join, itertools.product(*dim_names)))
    return names


def _jax_array():
    if IS_JAX_INSTALLED:
        entry = {
            jaxlib.xla_extension.DeviceArray: {
                "flatten": lambda arr: (arr.flatten().tolist(), arr.shape),
                "unflatten": lambda aux_data, leaves: jax.numpy.array(leaves).reshape(
                    aux_data
                ),
                "names": _array_element_names,
            },
        }
    else:
        entry = {}
    return entry


def _pandas_series():
    """Create registry entry for pandas.Series."""
    if IS_PANDAS_INSTALLED:
        entry = {
            pd.Series: {
                "flatten": lambda sr: (
                    sr.tolist(),
                    {"index": sr.index, "name": sr.name},
                ),
                "unflatten": lambda aux_data, leaves: pd.Series(leaves, **aux_data),
                "names": lambda sr: list(sr.index.map(_index_element_to_string)),
            },
        }
    else:
        entry = {}
    return entry


def _pandas_dataframe():
    """Create registry entry for pandas.DataFrame."""
    if IS_PANDAS_INSTALLED:
        entry = {
            pd.DataFrame: {
                "flatten": _flatten_pandas_dataframe,
                "unflatten": _unflatten_pandas_dataframe,
                "names": _get_names_pandas_dataframe,
            }
        }
    else:
        entry = {}
    return entry


def _flatten_pandas_dataframe(df):
    flat = df.to_numpy().flatten().tolist()
    aux_data = {"columns": df.columns, "index": df.index, "shape": df.shape}
    return flat, aux_data


def _unflatten_pandas_dataframe(aux_data, leaves):
    out = pd.DataFrame(
        data=np.array(leaves).reshape(aux_data["shape"]),
        columns=aux_data["columns"],
        index=aux_data["index"],
    )
    return out


def _get_names_pandas_dataframe(df):
    index_strings = list(df.index.map(_index_element_to_string))
    out = ["_".join([loc, col]) for loc, col in product(index_strings, df.columns)]
    return out


def _index_element_to_string(element):
    if isinstance(element, (tuple, list)):
        as_strings = [str(entry) for entry in element]
        res_string = "_".join(as_strings)
    else:
        res_string = str(element)

    return res_string


FUNC_DICT = {
    "list": _list,
    "tuple": _tuple,
    "dict": _dict,
    "numpy.ndarray": _numpy_array,
    "jax.numpy.ndarray": _jax_array,
    "pandas.Series": _pandas_series,
    "pandas.DataFrame": _pandas_dataframe,
    "None": _none,
    "namedtuple": _namedtuple,
    "OrderedDict": _ordereddict,
}
