from __future__ import annotations

import pytest

import construction_helpers.composition  # noqa: F401 - enables `@` composition
import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def _attention_term():
    qk_matmul = ops.Einops.template("q h d, x h d -> h q d")
    softmax = ops.SoftMax.template()
    mask = ops.WeightedTriangularLower.template()
    sv_matmul = ops.Einops.template("h q x, x h d -> q h d")
    return qk_matmul @ softmax @ mask @ sv_matmul


def test_attention_compiles_structurally() -> None:
    term = _attention_term()
    result = tc_compile(term)
    assert len(result.kernel_sources) == 4
    for src in result.kernel_sources:
        import ast
        ast.parse(src)


@pytest.mark.requires_gpu
def test_attention_matches_torch_compile() -> None:
    import torch

    from torch_compile.torch_compile import ConstructedModule

    term = _attention_term()
    triton_mod = tc_compile(term).to_module().cuda()
    torch_mod = ConstructedModule.construct(term).cuda()

    torch.manual_seed(0)
    q = torch.randn(8, 4, 16, device="cuda")
    k = torch.randn(8, 4, 16, device="cuda")
    v = torch.randn(8, 4, 16, device="cuda")

    y_triton = triton_mod(q, k, v)
    y_torch = torch_mod(q, k, v)
    assert torch.allclose(y_triton, y_torch, atol=1e-3, rtol=1e-3)


def test_attention_matches_golden() -> None:
    from pathlib import Path

    term = _attention_term()
    result = tc_compile(term)
    actual = ""
    for i, src in enumerate(result.kernel_sources):
        actual += f"# Kernel {i}\n{src}\n"
    expected = Path(__file__).parent.joinpath(
        "golden/attention_stageA.triton.py"
    ).read_text()
    assert actual == expected, (
        "Triton source drift. If intentional, regenerate the golden file."
    )
