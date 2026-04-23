from __future__ import annotations

from dataclasses import dataclass

import pytest

import data_structure.Category as cat
from triton_compile import operators as tops


@dataclass(frozen=True)
class _Unregistered(cat.Operator):
    pass


@dataclass(frozen=True)
class _Toy(cat.Operator):
    pass


def test_registry_has_no_entry_for_unregistered_type() -> None:
    assert _Unregistered not in tops.TritonOperator.registry


def test_subclass_registers_itself() -> None:
    class _ToyTriton(tops.TritonOperator, operation_key=_Toy):
        def emit(self, target: cat.Broadcasted) -> str:
            return "TOY"

    assert tops.TritonOperator.registry[_Toy] is _ToyTriton


def test_dispatch_unknown_operator_raises() -> None:
    with pytest.raises(NotImplementedError, match="No TritonOperator"):
        tops.TritonOperator.dispatch(_Unregistered())
