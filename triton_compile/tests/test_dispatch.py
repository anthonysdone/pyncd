from __future__ import annotations

import pytest

import construction_helpers.composition  # noqa: F401 - enables `@` composition
import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def test_compile_rejects_unregistered_broadcast() -> None:
    # Build a term using ReLU (not yet registered in Task 5 — Task 6 adds it).
    term = ops.ReLU.template()
    with pytest.raises(NotImplementedError, match="No TritonOperator"):
        tc_compile(term)


def test_compile_walks_composed() -> None:
    # A Composed of two unregistered broadcasts should still fail at the
    # leaf, not at the Composed — proves walk reaches leaves.
    a = ops.ReLU.template()
    b = ops.ReLU.template()
    composed = a @ b
    with pytest.raises(NotImplementedError, match="No TritonOperator"):
        tc_compile(composed)
