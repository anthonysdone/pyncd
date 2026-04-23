from __future__ import annotations

import ast

import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def test_embedding_emits_valid_triton() -> None:
    term = ops.Embedding.template(embedding_size="V", output_size=32)
    result = tc_compile(term)
    src = result.kernel_sources[0]
    ast.parse(src)
    assert "tl.load" in src
