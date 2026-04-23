from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Type

import data_structure.Category as cat


class TritonOperator(ABC):
    registry: ClassVar[dict[Type[cat.Operator], Type["TritonOperator"]]] = {}

    def __init_subclass__(cls, operation_key: Type[cat.Operator] | None = None) -> None:
        super().__init_subclass__()
        if operation_key is not None:
            TritonOperator.registry[operation_key] = cls

    @classmethod
    def dispatch(cls, operator: cat.Operator) -> "TritonOperator":
        op_type = type(operator)
        if op_type not in cls.registry:
            raise NotImplementedError(
                f"No TritonOperator registered for {op_type.__name__}"
            )
        return cls.registry[op_type]()

    @abstractmethod
    def emit(self, target: cat.Broadcasted) -> str:
        """Return Triton source for a single broadcasted-operator kernel."""


import data_structure.Operators as ops
from triton_compile import codegen


_ELEMENTWISE_OP: dict[type, str] = {
    ops.ReLU: "tl.maximum(x, 0.0)",
    ops.Elementwise: "tl.sigmoid(x)",
    ops.Dropout: "x",  # Stage A: no-op; dropout needs RNG state (deferred).
    ops.Identity: "x",
}


class _ElementwiseTriton(TritonOperator, operation_key=ops.Elementwise):
    def emit(self, target: cat.Broadcasted) -> str:
        op_type = type(target.operator)
        expr = _ELEMENTWISE_OP.get(op_type, "x")
        name = f"_elementwise_{op_type.__name__.lower()}_kernel"
        body = (
            "\n    pid = tl.program_id(0)"
            "\n    BLOCK: tl.constexpr = 1024"
            "\n    offsets = pid * BLOCK + tl.arange(0, BLOCK)"
            "\n    mask = offsets < n_elements"
            "\n    x = tl.load(x_ptr + offsets, mask=mask)"
            f"\n    tl.store(y_ptr + offsets, {expr}, mask=mask)"
        )
        return codegen.KernelSource(
            name=name,
            params=[
                codegen.Param("x_ptr", "pointer"),
                codegen.Param("y_ptr", "pointer"),
                codegen.Param("n_elements", "i32"),
            ],
            body=body,
        ).render()


class _ReLUTriton(_ElementwiseTriton, operation_key=ops.ReLU):
    pass


class _DropoutTriton(_ElementwiseTriton, operation_key=ops.Dropout):
    pass


class _IdentityTriton(_ElementwiseTriton, operation_key=ops.Identity):
    pass


class _AdditionTriton(TritonOperator, operation_key=ops.AdditionOp):
    def emit(self, target: cat.Broadcasted) -> str:
        body = (
            "\n    pid = tl.program_id(0)"
            "\n    BLOCK: tl.constexpr = 1024"
            "\n    offsets = pid * BLOCK + tl.arange(0, BLOCK)"
            "\n    mask = offsets < n_elements"
            "\n    x0 = tl.load(x0_ptr + offsets, mask=mask)"
            "\n    x1 = tl.load(x1_ptr + offsets, mask=mask)"
            "\n    tl.store(y_ptr + offsets, x0 + x1, mask=mask)"
        )
        return codegen.KernelSource(
            name="_addition_kernel",
            params=[
                codegen.Param("x0_ptr", "pointer"),
                codegen.Param("x1_ptr", "pointer"),
                codegen.Param("y_ptr", "pointer"),
                codegen.Param("n_elements", "i32"),
            ],
            body=body,
        ).render()


def _has_contraction(op: ops.Einops) -> bool:
    # A contraction occurs in multi-input operations where axes are shared/reduced.
    # Single-input operations (len(signature) == 1) are rearrangements with no contraction.
    # Multi-input operations with shared axes (non-empty intersection) have contractions.
    return len(op.signature) > 1


def _emit_einops_rearrange(target: cat.Broadcasted) -> str:
    body = (
        "\n    pid = tl.program_id(0)"
        "\n    BLOCK: tl.constexpr = 1024"
        "\n    offsets = pid * BLOCK + tl.arange(0, BLOCK)"
        "\n    mask = offsets < n_elements"
        "\n    x = tl.load(x_ptr + offsets, mask=mask)"
        "\n    tl.store(y_ptr + offsets, x, mask=mask)"
    )
    return codegen.KernelSource(
        name="_einops_rearrange_kernel",
        params=[
            codegen.Param("x_ptr", "pointer"),
            codegen.Param("y_ptr", "pointer"),
            codegen.Param("n_elements", "i32"),
        ],
        body=body,
    ).render()


def _emit_einops_matmul(target: cat.Broadcasted) -> str:
    body = (
        "\n    pid_m = tl.program_id(0)"
        "\n    pid_n = tl.program_id(1)"
        "\n    BLOCK_M: tl.constexpr = 64"
        "\n    BLOCK_N: tl.constexpr = 64"
        "\n    BLOCK_K: tl.constexpr = 32"
        "\n    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)"
        "\n    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)"
        "\n    offs_k = tl.arange(0, BLOCK_K)"
        "\n    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)"
        "\n    for k in range(0, K, BLOCK_K):"
        "\n        a_ptrs = a_ptr + (offs_m[:, None] * K + (k + offs_k)[None, :])"
        "\n        b_ptrs = b_ptr + ((k + offs_k)[:, None] * N + offs_n[None, :])"
        "\n        a_mask = (offs_m[:, None] < M) & ((k + offs_k)[None, :] < K)"
        "\n        b_mask = ((k + offs_k)[:, None] < K) & (offs_n[None, :] < N)"
        "\n        a = tl.load(a_ptrs, mask=a_mask, other=0.0)"
        "\n        b = tl.load(b_ptrs, mask=b_mask, other=0.0)"
        "\n        acc += tl.dot(a, b)"
        "\n    c_ptrs = c_ptr + (offs_m[:, None] * N + offs_n[None, :])"
        "\n    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)"
        "\n    tl.store(c_ptrs, acc, mask=c_mask)"
    )
    return codegen.KernelSource(
        name="_einops_matmul_kernel",
        params=[
            codegen.Param("a_ptr", "pointer"),
            codegen.Param("b_ptr", "pointer"),
            codegen.Param("c_ptr", "pointer"),
            codegen.Param("M", "i32"),
            codegen.Param("N", "i32"),
            codegen.Param("K", "i32"),
        ],
        body=body,
    ).render()


class _EinopsTriton(TritonOperator, operation_key=ops.Einops):
    def emit(self, target: cat.Broadcasted) -> str:
        op: ops.Einops = target.operator  # type: ignore[assignment]
        if _has_contraction(op):
            return _emit_einops_matmul(target)
        return _emit_einops_rearrange(target)
