from collections import namedtuple
from typing import NamedTuple

from pybaum.typecheck import get_type


def test_namedtuple_is_discovered():
    bla = namedtuple("bla", ["a", "b"])(1, 2)
    assert get_type(bla) == namedtuple


def test_typed_namedtuple_is_discovered():
    class Blubb(NamedTuple):
        a: int
        b: int

    blubb = Blubb(1, 2)
    assert get_type(blubb) == namedtuple


def test_standard_tuple_is_not_discovered():
    assert get_type((1, 2)) == tuple
