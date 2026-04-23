from __future__ import annotations

import ast

from triton_compile import codegen


def test_kernel_source_is_valid_python() -> None:
    src = codegen.KernelSource(
        name="_relu_kernel",
        params=[
            codegen.Param("x_ptr", "pointer"),
            codegen.Param("y_ptr", "pointer"),
            codegen.Param("n_elements", "i32"),
            codegen.Param("BLOCK", "constexpr"),
        ],
        body="""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, tl.maximum(x, 0.0), mask=mask)
""",
    ).render()
    ast.parse(src)
    assert "@triton.jit" in src
    assert "def _relu_kernel(" in src
    assert "BLOCK: tl.constexpr" in src


def test_grid_1d_computes_cdiv() -> None:
    grid = codegen.grid_1d(total=1024, block=128)
    assert grid == (8,)


def test_grid_1d_rounds_up() -> None:
    grid = codegen.grid_1d(total=1000, block=128)
    assert grid == (8,)


def test_grid_2d() -> None:
    grid = codegen.grid_2d(total=(1024, 64), block=(128, 32))
    assert grid == (8, 2)
