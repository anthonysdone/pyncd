from __future__ import annotations

import ast

import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def test_normalize_emits_valid_triton() -> None:
    term = ops.Normalize.template(input_size=64)
    result = tc_compile(term)
    src = result.kernel_sources[0]
    ast.parse(src)
    assert "tl.sum" in src
    assert "tl.rsqrt" in src
