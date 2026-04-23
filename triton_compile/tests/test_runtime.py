from __future__ import annotations

import pytest

import data_structure.Operators as ops
from triton_compile import TritonModule, compile as tc_compile


def test_compile_returns_triton_module() -> None:
    term = ops.ReLU.template()
    mod = tc_compile(term).to_module()
    assert isinstance(mod, TritonModule)


def test_module_has_kernel_sources() -> None:
    term = ops.ReLU.template()
    mod = tc_compile(term).to_module()
    assert len(mod.kernel_sources) == 1
    assert "@triton.jit" in mod.kernel_sources[0]


@pytest.mark.requires_gpu
def test_module_forward_relu(gpu_available: bool) -> None:
    import torch

    term = ops.ReLU.template()
    mod = tc_compile(term).to_module().cuda()
    x = torch.randn(1024, device="cuda")
    y = mod(x)
    assert torch.allclose(y, torch.relu(x), atol=1e-4, rtol=1e-4)
