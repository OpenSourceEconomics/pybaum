pybaum
======

.. start-badges

.. image:: https://img.shields.io/pypi/v/pybaum?color=blue
    :alt: PyPI
    :target: https://pypi.org/project/pybaum

.. image:: https://img.shields.io/pypi/pyversions/pybaum
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/pybaum

.. image:: https://img.shields.io/conda/vn/conda-forge/pybaum.svg
    :target: https://anaconda.org/conda-forge/pybaum

.. image:: https://img.shields.io/conda/pn/conda-forge/pybaum.svg
    :target: https://anaconda.org/conda-forge/pybaum

.. image:: https://img.shields.io/pypi/l/pybaum
    :alt: PyPI - License
    :target: https://pypi.org/project/pybaum

.. image:: https://readthedocs.org/projects/pybaum/badge/?version=latest
    :target: https://pybaum.readthedocs.io/en/latest

.. image:: https://img.shields.io/github/workflow/status/OpenSourceEconomics/pybaum/main/main
   :target: https://github.com/OpenSourceEconomics/pybaum/actions?query=branch%3Amain

.. image:: https://codecov.io/gh/OpenSourceEconomics/pybaum/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/OpenSourceEconomics/pybaum

.. image:: https://results.pre-commit.ci/badge/github/OpenSourceEconomics/pybaum/main.svg
    :target: https://results.pre-commit.ci/latest/github/OpenSourceEconomics/pybaum/main
    :alt: pre-commit.ci status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

.. end-badges

Installation
------------

pybaum is available on `PyPI <https://pypi.org/project/pybaum>`_ and `Anaconda.org
<https://anaconda.org/conda-forge/pybaum>`_. Install it with

.. code-block:: console

    $ pip install pybaum

    # or

    $ conda install -c conda-forge pybaum


About
-----

pybaum provides tools to work with pytrees which is a concept borrowed from `JAX
<https://jax.readthedocs.io/en/latest/>`_.

What is a pytree?

In pybaum, we use the term pytree to refer to a tree-like structure built out of
container-like Python objects. Classes are considered container-like if they are in the
pytree registry, which by default includes lists, tuples, and dicts. That is:

1. Any object whose type is not in the pytree container registry is considered a leaf
   pytree.

2. Any object whose type is in the pytree container registry, and which contains
   pytrees, is considered a pytree.

For each entry in the pytree container registry, a container-like type is registered
with a pair of functions that specify how to convert an instance of the container type
to a (children, metadata) pair and how to convert such a pair back to an instance of the
container type. Using these functions, pybaum can canonicalize any tree of registered
container objects into tuples.

Example pytrees:

.. code-block:: python

    [1, "a", object()]  # 3 leaves

    (1, (2, 3), ())  # 3 leaves

    [1, {"k1": 2, "k2": (3, 4)}, 5]  # 5 leaves

pybaum can be extended to consider other container types as pytrees.
