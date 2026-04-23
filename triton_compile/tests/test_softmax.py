from __future__ import annotations

import ast

import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def test_softmax_emits_valid_triton() -> None:
    term = ops.SoftMax.template()
    result = tc_compile(term)
    assert len(result.kernel_sources) == 1
    src = result.kernel_sources[0]
    ast.parse(src)
    assert "tl.exp" in src
    assert "tl.max" in src
    assert "tl.sum" in src
