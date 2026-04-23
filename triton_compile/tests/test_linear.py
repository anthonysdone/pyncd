from __future__ import annotations

import ast
from dataclasses import replace

import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def test_linear_emits_valid_triton() -> None:
    term = ops.Linear.template(input_size=32, output_size=16)
    result = tc_compile(term)
    src = result.kernel_sources[0]
    ast.parse(src)
    assert "tl.dot" in src


def test_linear_with_bias() -> None:
    term = ops.Linear.template(input_size=32, output_size=16)
    biased_op = replace(term.operator, bias=True)
    biased = replace(term, operator=biased_op)
    result = tc_compile(biased)
    src = result.kernel_sources[0]
    assert "bias_ptr" in src
