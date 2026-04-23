from __future__ import annotations

import pytest


def test_marker_registered(pytestconfig: pytest.Config) -> None:
    markers = pytestconfig.getini("markers")
    assert any("requires_gpu" in m for m in markers)


@pytest.mark.requires_gpu
def test_requires_gpu_skips_on_mac(gpu_available: bool) -> None:
    assert gpu_available, "this test should not run without a GPU"


def test_gpu_fixture_returns_bool(gpu_available: bool) -> None:
    assert isinstance(gpu_available, bool)
