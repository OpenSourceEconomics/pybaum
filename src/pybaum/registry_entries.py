import itertools
from functools import partial

from pybaum.config import IS_NUMPY_INSTALLED
from pybaum.config import IS_PANDAS_INSTALLED

if IS_NUMPY_INSTALLED:
    import numpy as np

if IS_PANDAS_INSTALLED:
    import pandas as pd


def _list():
    entry = {
        list: {
            "flatten": lambda tree: (tree, None),
            "unflatten": lambda aux_data, children: children,  # noqa: U100
            "names": lambda tree: [f"{i}" for i in range(len(tree))],
        },
    }
    return entry


def _dict():
    entry = {
        dict: {
            "flatten": lambda tree: (list(tree.values()), list(tree)),
            "unflatten": lambda aux_data, children: dict(zip(aux_data, children)),
            "names": lambda tree: list(map(str, list(tree))),
        },
    }
    return entry


def _tuple():
    entry = {
        tuple: {
            "flatten": lambda tree: (list(tree), None),
            "unflatten": lambda aux_data, children: tuple(children),  # noqa: U100
            "names": lambda tree: [f"{i}" for i in range(len(tree))],
        },
    }
    return entry


def _numpy_array():
    """Create a pytree declaration for numpy arrays.

    To-Do: Add optional axis argument.

    """
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


def _pandas_series():
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


def _pandas_dataframe(columns=None):
    if IS_PANDAS_INSTALLED:
        entry = {
            pd.DataFrame: {
                "flatten": partial(_flatten_pandas_dataframe, columns=columns),
                "unflatten": partial(_unflatten_pandas_dataframe),
                "names": partial(_get_names_pandas_dataframe, columns=columns),
            }
        }
    else:
        entry = {}
    return entry


def _flatten_pandas_dataframe(df, columns):
    columns = _process_columns(df, columns)
    flat = []
    for col in columns:
        flat += df[col].tolist()

    aux_data = (columns, df.drop(columns=columns))
    return flat, aux_data


def _unflatten_pandas_dataframe(aux_data, leaves):
    columns, empty_df = aux_data
    out = empty_df.copy()
    remaining_leaves = leaves
    for col in columns:
        out[col] = leaves[: len(empty_df)]
        remaining_leaves = remaining_leaves[len(empty_df) :]
    return out


def _get_names_pandas_dataframe(df, columns):
    columns = _process_columns(df, columns)
    if len(columns) == 1:
        out = list(df.index.map(_index_element_to_string))
    else:
        out = []
        for col in df.columns:
            out += list(df.index.map(partial(_index_element_to_string, prefix=col)))
    return out


def _process_columns(df, columns):
    if columns is None:
        columns = df.columns
    elif not isinstance(columns, list):
        columns = [columns]
    return columns


def _index_element_to_string(element, prefix=None):
    separator = "_"
    if isinstance(element, (tuple, list)):
        as_strings = [str(entry) for entry in element]
        res_string = separator.join(as_strings)
    else:
        res_string = str(element)

    if prefix is not None:
        res_string = separator.join([prefix, res_string])
    return res_string


FUNC_DICT = {
    "list": _list,
    "tuple": _tuple,
    "dict": _dict,
    "numpy.ndarray": _numpy_array,
    "pandas.Series": _pandas_series,
    "pandas.DataFrame": _pandas_dataframe,
}
