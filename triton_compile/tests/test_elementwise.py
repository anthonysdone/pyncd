from __future__ import annotations

import ast

import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def test_relu_emits_valid_triton() -> None:
    term = ops.ReLU.template()
    result = tc_compile(term)
    assert len(result.kernel_sources) == 1
    src = result.kernel_sources[0]
    ast.parse(src)
    assert "@triton.jit" in src
    assert "tl.maximum" in src


def test_relu_kernel_name_is_deterministic() -> None:
    term = ops.ReLU.template()
    result_a = tc_compile(term)
    result_b = tc_compile(term)
    assert result_a.kernel_sources == result_b.kernel_sources
