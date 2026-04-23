from __future__ import annotations

import ast

import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def test_addition_emits_valid_triton() -> None:
    # AdditionOp.template asserts all signature segments are () — use default.
    term = ops.AdditionOp.template()
    result = tc_compile(term)
    assert len(result.kernel_sources) == 1
    src = result.kernel_sources[0]
    ast.parse(src)
    assert "tl.load(x0_ptr" in src
    assert "tl.load(x1_ptr" in src
    assert " + " in src
