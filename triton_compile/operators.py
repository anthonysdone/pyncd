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
