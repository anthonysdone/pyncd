from __future__ import annotations

import data_structure.Category as cat
from triton_compile import operators as tops


def compile(term: cat.Morphism) -> "CompiledTerm":
    """Walk the term and produce a list of kernel sources.

    Stage A: one kernel per Broadcasted. Rearrangements fold into
    address math (emit nothing). Composed/Product/Block are structural.
    """
    kernels: list[str] = []
    _walk(term, kernels)
    return CompiledTerm(term=term, kernel_sources=tuple(kernels))


def _walk(term: cat.Morphism, out: list[str]) -> None:
    match term:
        case cat.Rearrangement():
            return
        case cat.ProductOfMorphisms():
            for sub in term.content:
                _walk(sub, out)
        case cat.Composed():
            for sub in term.content:
                _walk(sub, out)
        case cat.Block():
            _walk(term.body, out)
        case cat.Broadcasted():
            op = tops.TritonOperator.dispatch(term.operator)
            out.append(op.emit(term))
        case _:
            raise NotImplementedError(f"Unhandled term type: {type(term).__name__}")


from dataclasses import dataclass


@dataclass(frozen=True)
class CompiledTerm:
    term: cat.Morphism
    kernel_sources: tuple[str, ...]

    def to_module(self) -> "TritonModule":
        from triton_compile.runtime import TritonModule

        return TritonModule(self.term, self.kernel_sources)
