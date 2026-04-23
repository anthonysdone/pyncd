from __future__ import annotations

import pytest

import data_structure.Operators as ops
from triton_compile import compile as tc_compile


@pytest.mark.requires_gpu
def test_launch_single_relu() -> None:
    import torch

    term = ops.ReLU.template()
    mod = tc_compile(term).to_module().cuda()
    x = torch.randn(1024, device="cuda")
    y = mod(x)
    assert torch.allclose(y, torch.relu(x), atol=1e-4)


@pytest.mark.requires_gpu
def test_launch_composed_relu_relu() -> None:
    import torch

    import construction_helpers.composition  # noqa: F401

    term = ops.ReLU.template() @ ops.ReLU.template()
    mod = tc_compile(term).to_module().cuda()
    x = torch.randn(1024, device="cuda")
    y = mod(x)
    assert torch.allclose(y, torch.relu(torch.relu(x)), atol=1e-4)
