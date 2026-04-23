from __future__ import annotations


def test_package_imports() -> None:
    import triton_compile

    assert hasattr(triton_compile, "__all__")


def test_triton_compile_module_exists() -> None:
    from triton_compile import triton_compile as tc

    assert tc is not None
