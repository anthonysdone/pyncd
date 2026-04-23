from __future__ import annotations

from dataclasses import dataclass

import pytest

import construction_helpers.composition  # noqa: F401 - enables `@` composition
import data_structure.Category as cat
import data_structure.Operators as ops
from data_structure.Operators import sized
from triton_compile import compile as tc_compile


@dataclass(frozen=True)
class _UnregisteredDispatch(cat.Operator):
    pass


def test_compile_rejects_unregistered_broadcast() -> None:
    term = sized(_UnregisteredDispatch(), input_size=1)
    with pytest.raises(NotImplementedError, match="No TritonOperator"):
        tc_compile(term)


def test_compile_walks_composed() -> None:
    # ReLU is registered — confirm a composed term compiles successfully
    # (walk reaches both leaves without error).
    a = ops.ReLU.template()
    b = ops.ReLU.template()
    composed = a @ b
    result = tc_compile(composed)
    assert len(result.kernel_sources) == 2
