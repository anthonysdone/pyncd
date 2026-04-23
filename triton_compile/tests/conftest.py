from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    import torch

    if torch.cuda.is_available():
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA GPU available")
    for item in items:
        if "requires_gpu" in item.keywords:
            item.add_marker(skip_gpu)


@pytest.fixture
def gpu_available() -> bool:
    import torch

    return torch.cuda.is_available()
