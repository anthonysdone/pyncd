from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal

ParamKind = Literal["pointer", "i32", "i64", "constexpr"]


@dataclass(frozen=True)
class Param:
    name: str
    kind: ParamKind

    def render(self) -> str:
        match self.kind:
            case "pointer":
                return self.name
            case "i32":
                return f"{self.name}: tl.int32"
            case "i64":
                return f"{self.name}: tl.int64"
            case "constexpr":
                return f"{self.name}: tl.constexpr"


@dataclass(frozen=True)
class KernelSource:
    name: str
    params: list[Param]
    body: str

    def render(self) -> str:
        sig = ", ".join(p.render() for p in self.params)
        return (
            "import triton\n"
            "import triton.language as tl\n"
            "\n"
            "@triton.jit\n"
            f"def {self.name}({sig}):\n"
            f"{self.body.rstrip()}\n"
        )


def grid_1d(total: int, block: int) -> tuple[int]:
    return (ceil(total / block),)


def grid_2d(total: tuple[int, int], block: tuple[int, int]) -> tuple[int, int]:
    return (ceil(total[0] / block[0]), ceil(total[1] / block[1]))
