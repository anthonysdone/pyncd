from __future__ import annotations

import ast

import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def test_einops_no_contraction_emits_valid_triton() -> None:
    # 'a b -> b a' is a transpose: no contraction, just rearrange.
    term = ops.Einops.template("a b -> b a")
    result = tc_compile(term)
    assert len(result.kernel_sources) == 1
    src = result.kernel_sources[0]
    ast.parse(src)
    assert "tl.load" in src
    assert "tl.store" in src


def test_einops_contraction_emits_matmul() -> None:
    # 'i k, k j -> i j' is a matrix multiply.
    term = ops.Einops.template("i k, k j -> i j")
    result = tc_compile(term)
    assert len(result.kernel_sources) == 1
    src = result.kernel_sources[0]
    ast.parse(src)
    assert "tl.dot" in src
