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
