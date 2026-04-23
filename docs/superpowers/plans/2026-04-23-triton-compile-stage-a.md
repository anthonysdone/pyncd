# Triton Compile — Stage A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a `triton_compile/` package that compiles pyncd `BroadcastedCategory` terms into runnable Triton kernels, one kernel per `Broadcasted` (no cross-op fusion). Mirrors `torch_compile/` in structure and API.

**Architecture:** Registry-dispatched compiler. A top-level `compile(term)` walks `Composed`/`ProductOfMorphisms`/`Block`/`Rearrangement` structurally and dispatches each `Broadcasted` on its operator type. Each operator has a `TritonOperator` subclass that owns a Triton-source template and a launch spec. The result is a `TritonModule` (`nn.Module`) that owns the compiled `@triton.jit` kernels and runs them sequentially.

**Tech Stack:** Python 3.14, `triton` (≥3.0), `torch`, `einops`, `pytest`. Reference: `torch_compile/torch_compile.py` and the spec at `docs/superpowers/specs/2026-04-23-triton-kernel-generation-design.md`.

**Planning notes for the implementing engineer:**
- The Mac phase can't run Triton kernels numerically. Correctness tests are marked `@requires_gpu` and skip on Mac. Structural tests (`ast.parse` of emitted source, grid/shape computations) always run and are the core Mac-phase signal.
- Every `triton_compile/` module MUST start with `from __future__ import annotations`. Follow the existing pyncd style: `@dataclass(frozen=True)` for value objects, `TypeVar` with defaults for generics, no comments unless the *why* is non-obvious.
- Reference `torch_compile/torch_compile.py` for the dispatcher pattern — we are mirroring it. Read that file before starting Task 4.
- Commit after each green test. Do not batch commits.

**Pre-flight: read these before starting.**
1. Spec: `docs/superpowers/specs/2026-04-23-triton-kernel-generation-design.md` (sections 4–6, 9).
2. Reference compiler: `torch_compile/torch_compile.py` — this is the pattern we mirror.
3. Operator definitions: `data_structure/Operators.py`.
4. Kernel primitives already in the repo: `data_structure_kernels/Kernel.py`.

---

## File structure

Create in `triton_compile/`:

| File | Responsibility |
| ---- | -------------- |
| `__init__.py` | Public exports: `compile`, `TritonModule`. |
| `triton_compile.py` | Top-level `compile(term)` entry + structural dispatch over `Composed`/`Product`/`Block`/`Rearrangement`. Parallel to `torch_compile.py:construct`. |
| `operators.py` | `TritonOperator` base class with operator registry; one subclass per pyncd operator. |
| `codegen.py` | Pure-Python Triton source builder: kernel signature generation, block-pointer emission, launch-grid computation. No Triton import; so it's testable on Mac without Triton installed. |
| `runtime.py` | `TritonModule(nn.Module)` that owns compiled kernels and sequences launches. |
| `tests/__init__.py` | empty |
| `tests/conftest.py` | `@requires_gpu` marker, golden-file helpers, fixture factories for toy inputs. |
| `tests/test_codegen.py` | Layer 1 (structural): `ast.parse` round-trip, shape/grid correctness. |
| `tests/test_operators.py` | Per-operator: structural assertion the kernel source is valid + GPU correctness (marked). |
| `tests/test_integration.py` | End-to-end: attention, transformer block. |
| `tests/golden/` | Frozen source snapshots; one file per canonical model. |

Out of scope for Stage A (no files): `fusion.py`, `rewrites.py`, `cost_model.py`, `kernel_seeds.py`, `hardware/`.

---

## Task 1: Package scaffolding + dependencies

**Files:**
- Create: `triton_compile/__init__.py`
- Create: `triton_compile/tests/__init__.py`
- Modify: `requirements.txt`
- Create: `triton_compile/tests/test_smoke.py`

- [ ] **Step 1: Add triton to requirements**

Modify `requirements.txt` — append these lines (leave existing lines untouched):

```
triton>=3.0
pytest>=8.0
```

- [ ] **Step 2: Create empty package files**

Create `triton_compile/__init__.py` with contents:

```python
from __future__ import annotations

__all__: list[str] = []
```

Create `triton_compile/tests/__init__.py` with empty contents (one blank line).

- [ ] **Step 3: Write failing smoke test**

Create `triton_compile/tests/test_smoke.py`:

```python
from __future__ import annotations


def test_package_imports() -> None:
    import triton_compile

    assert hasattr(triton_compile, "__all__")


def test_triton_compile_module_exists() -> None:
    from triton_compile import triton_compile as tc

    assert tc is not None
```

- [ ] **Step 4: Run the test and confirm failure**

Run: `pytest triton_compile/tests/test_smoke.py -v`
Expected: `test_package_imports` PASS, `test_triton_compile_module_exists` FAIL with `ModuleNotFoundError`.

- [ ] **Step 5: Create the empty module**

Create `triton_compile/triton_compile.py`:

```python
from __future__ import annotations
```

- [ ] **Step 6: Run the test and confirm pass**

Run: `pytest triton_compile/tests/test_smoke.py -v`
Expected: both PASS.

- [ ] **Step 7: Commit**

```bash
git add triton_compile/__init__.py triton_compile/tests/__init__.py \
        triton_compile/tests/test_smoke.py triton_compile/triton_compile.py \
        requirements.txt
git commit -m "feat(triton_compile): scaffold package"
```

---

## Task 2: Test infrastructure — conftest and GPU marker

**Files:**
- Create: `triton_compile/tests/conftest.py`
- Modify: `pyproject.toml` OR create `pytest.ini` (see Step 1)
- Create: `triton_compile/tests/test_conftest.py`

- [ ] **Step 1: Register the `requires_gpu` marker**

Check whether `pyproject.toml` exists at repo root. If it does, append under a `[tool.pytest.ini_options]` section; if not, create `pytest.ini`. Use `pytest.ini` if in doubt.

Create `pytest.ini` at repo root:

```ini
[pytest]
markers =
    requires_gpu: marks tests that require a CUDA GPU (skipped on Mac/CPU)
testpaths = triton_compile/tests
```

- [ ] **Step 2: Write conftest with the skip logic**

Create `triton_compile/tests/conftest.py`:

```python
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
```

- [ ] **Step 3: Write a failing test that exercises the marker**

Create `triton_compile/tests/test_conftest.py`:

```python
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
```

- [ ] **Step 4: Run and confirm**

Run: `pytest triton_compile/tests/test_conftest.py -v`
Expected on Mac: `test_marker_registered` PASS, `test_requires_gpu_skips_on_mac` SKIPPED (reason: "no CUDA GPU available"), `test_gpu_fixture_returns_bool` PASS.

- [ ] **Step 5: Commit**

```bash
git add pytest.ini triton_compile/tests/conftest.py \
        triton_compile/tests/test_conftest.py
git commit -m "test(triton_compile): add requires_gpu marker and conftest"
```

---

## Task 3: Codegen primitives — KernelSource builder

**Files:**
- Create: `triton_compile/codegen.py`
- Create: `triton_compile/tests/test_codegen.py`

The codegen module is **pure Python** — it produces Triton source as a string, doesn't import `triton`. This lets us unit-test emission on Mac without the runtime.

- [ ] **Step 1: Write the failing test first**

Create `triton_compile/tests/test_codegen.py`:

```python
from __future__ import annotations

import ast

from triton_compile import codegen


def test_kernel_source_is_valid_python() -> None:
    src = codegen.KernelSource(
        name="_relu_kernel",
        params=[
            codegen.Param("x_ptr", "pointer"),
            codegen.Param("y_ptr", "pointer"),
            codegen.Param("n_elements", "i32"),
            codegen.Param("BLOCK", "constexpr"),
        ],
        body="""
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, tl.maximum(x, 0.0), mask=mask)
""",
    ).render()
    ast.parse(src)
    assert "@triton.jit" in src
    assert "def _relu_kernel(" in src
    assert "BLOCK: tl.constexpr" in src


def test_grid_1d_computes_cdiv() -> None:
    grid = codegen.grid_1d(total=1024, block=128)
    assert grid == (8,)


def test_grid_1d_rounds_up() -> None:
    grid = codegen.grid_1d(total=1000, block=128)
    assert grid == (8,)


def test_grid_2d() -> None:
    grid = codegen.grid_2d(total=(1024, 64), block=(128, 32))
    assert grid == (8, 2)
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest triton_compile/tests/test_codegen.py -v`
Expected: ImportError on `triton_compile.codegen`.

- [ ] **Step 3: Implement codegen**

Create `triton_compile/codegen.py`:

```python
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
```

- [ ] **Step 4: Run and confirm pass**

Run: `pytest triton_compile/tests/test_codegen.py -v`
Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add triton_compile/codegen.py triton_compile/tests/test_codegen.py
git commit -m "feat(triton_compile): add codegen primitives and kernel source builder"
```

---

## Task 4: TritonOperator base class + registry

**Files:**
- Create: `triton_compile/operators.py`
- Create: `triton_compile/tests/test_operators.py`

Mirrors `torch_compile/torch_compile.py`'s `ConstructedModule.operation_registry` pattern. Each concrete operator subclass registers itself via `__init_subclass__`.

- [ ] **Step 1: Write the failing test**

Create `triton_compile/tests/test_operators.py`:

```python
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
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest triton_compile/tests/test_operators.py -v`
Expected: ImportError on `triton_compile.operators`.

- [ ] **Step 3: Implement operators module**

Create `triton_compile/operators.py`:

```python
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
```

- [ ] **Step 4: Run and confirm pass**

Run: `pytest triton_compile/tests/test_operators.py -v`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add triton_compile/operators.py triton_compile/tests/test_operators.py
git commit -m "feat(triton_compile): add TritonOperator base class and registry"
```

---

## Task 5: Top-level compile() — structural dispatch

**Files:**
- Modify: `triton_compile/triton_compile.py`
- Modify: `triton_compile/__init__.py`
- Create: `triton_compile/tests/test_dispatch.py`

This implements the `Composed`/`Product`/`Block`/`Rearrangement`/`Broadcasted` walk, mirroring `torch_compile.ConstructedModule.construct`. Broadcasted dispatch calls `TritonOperator.dispatch` from Task 4.

- [ ] **Step 1: Write the failing test**

Create `triton_compile/tests/test_dispatch.py`:

```python
from __future__ import annotations

import pytest

import construction_helpers.composition  # noqa: F401 - enables `@` composition
import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def test_compile_rejects_unregistered_broadcast() -> None:
    # Build a term using ReLU (not yet registered in Task 5 — Task 6 adds it).
    term = ops.ReLU.template()
    with pytest.raises(NotImplementedError, match="No TritonOperator"):
        tc_compile(term)


def test_compile_walks_composed() -> None:
    # A Composed of two unregistered broadcasts should still fail at the
    # leaf, not at the Composed — proves walk reaches leaves.
    a = ops.ReLU.template()
    b = ops.ReLU.template()
    composed = a @ b
    with pytest.raises(NotImplementedError, match="No TritonOperator"):
        tc_compile(composed)
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest triton_compile/tests/test_dispatch.py -v`
Expected: ImportError on `triton_compile.compile`.

- [ ] **Step 3: Implement compile()**

Rewrite `triton_compile/triton_compile.py`:

```python
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
```

- [ ] **Step 4: Update `__init__.py` to export compile**

Replace `triton_compile/__init__.py`:

```python
from __future__ import annotations

from triton_compile.triton_compile import compile, CompiledTerm

__all__ = ["compile", "CompiledTerm"]
```

- [ ] **Step 5: Run and confirm pass**

Run: `pytest triton_compile/tests/test_dispatch.py -v`
Expected: both PASS.

- [ ] **Step 6: Commit**

```bash
git add triton_compile/triton_compile.py triton_compile/__init__.py \
        triton_compile/tests/test_dispatch.py
git commit -m "feat(triton_compile): top-level compile with structural dispatch"
```

---

## Task 6: Elementwise operator (ReLU) — first concrete op

**Files:**
- Modify: `triton_compile/operators.py`
- Create: `triton_compile/tests/test_elementwise.py`

`Elementwise`, `ReLU`, `Dropout` all subclass `Elementwise` in pyncd. For Stage A we handle them with one template that dispatches on the function name.

- [ ] **Step 1: Write the failing test**

Create `triton_compile/tests/test_elementwise.py`:

```python
from __future__ import annotations

import ast

import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def test_relu_emits_valid_triton() -> None:
    term = ops.ReLU.template()
    result = tc_compile(term)
    assert len(result.kernel_sources) == 1
    src = result.kernel_sources[0]
    ast.parse(src)
    assert "@triton.jit" in src
    assert "tl.maximum" in src


def test_relu_kernel_name_is_deterministic() -> None:
    term = ops.ReLU.template()
    result_a = tc_compile(term)
    result_b = tc_compile(term)
    assert result_a.kernel_sources == result_b.kernel_sources
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest triton_compile/tests/test_elementwise.py -v`
Expected: FAIL with `NotImplementedError: No TritonOperator registered for ReLU`.

- [ ] **Step 3: Implement the Elementwise operator**

Append to `triton_compile/operators.py`:

```python
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
```

- [ ] **Step 4: Run and confirm pass**

Run: `pytest triton_compile/tests/test_elementwise.py -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add triton_compile/operators.py triton_compile/tests/test_elementwise.py
git commit -m "feat(triton_compile): elementwise operator (ReLU, Dropout, Identity)"
```

---

## Task 7: AdditionOp operator

**Files:**
- Modify: `triton_compile/operators.py`
- Create: `triton_compile/tests/test_addition.py`

- [ ] **Step 1: Write the failing test**

Create `triton_compile/tests/test_addition.py`:

```python
from __future__ import annotations

import ast

import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def test_addition_emits_valid_triton() -> None:
    # AdditionOp.template asserts all signature segments are () — use default.
    term = ops.AdditionOp.template()
    result = tc_compile(term)
    assert len(result.kernel_sources) == 1
    src = result.kernel_sources[0]
    ast.parse(src)
    assert "tl.load(x0_ptr" in src
    assert "tl.load(x1_ptr" in src
    assert " + " in src
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest triton_compile/tests/test_addition.py -v`
Expected: FAIL with `NotImplementedError: No TritonOperator registered for AdditionOp`.

- [ ] **Step 3: Implement AdditionOp**

Append to `triton_compile/operators.py`:

```python
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
```

- [ ] **Step 4: Run and confirm pass**

Run: `pytest triton_compile/tests/test_addition.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add triton_compile/operators.py triton_compile/tests/test_addition.py
git commit -m "feat(triton_compile): AdditionOp operator"
```

---

## Task 8: Einops operator (no-contraction path)

**Files:**
- Modify: `triton_compile/operators.py`
- Create: `triton_compile/tests/test_einops.py`

Einops without contraction is pure axis rearrangement + broadcast. We emit a load-store kernel with strides. Einops **with** contraction is Task 9 — a distinct codepath.

- [ ] **Step 1: Write the failing test for the no-contraction path**

Create `triton_compile/tests/test_einops.py`:

```python
from __future__ import annotations

import ast

import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def test_einops_no_contraction_emits_valid_triton() -> None:
    # 'a b -> b a' is a transpose: no contraction, just rearrange.
    term = ops.Einops.template("a b -> b a")
    result = tc_compile(term)
    assert len(result.kernel_sources) == 1
    src = result.kernel_sources[0]
    ast.parse(src)
    assert "tl.load" in src
    assert "tl.store" in src


def test_einops_contraction_emits_matmul() -> None:
    # 'i k, k j -> i j' is a matrix multiply.
    term = ops.Einops.template("i k, k j -> i j")
    result = tc_compile(term)
    assert len(result.kernel_sources) == 1
    src = result.kernel_sources[0]
    ast.parse(src)
    assert "tl.dot" in src
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest triton_compile/tests/test_einops.py -v`
Expected: FAIL with `NotImplementedError: No TritonOperator registered for Einops`.

- [ ] **Step 3: Implement Einops (detection of contraction)**

Detection rule: a signature has a contraction iff any input axis-name is absent from the output. We reuse pyncd's `operator.signature` — a `SignatureSegment` structure — rather than re-parsing strings.

Append to `triton_compile/operators.py`:

```python
class _EinopsTriton(TritonOperator, operation_key=ops.Einops):
    def emit(self, target: cat.Broadcasted) -> str:
        op: ops.Einops = target.operator  # type: ignore[assignment]
        if _has_contraction(op):
            return _emit_einops_matmul(target)
        return _emit_einops_rearrange(target)


def _has_contraction(op: ops.Einops) -> bool:
    input_axes: set[int] = set().union(*op.signature[:-1]) if op.signature else set()
    output_axes: set[int] = set(op.signature[-1]) if op.signature else set()
    return bool(input_axes - output_axes)


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
```

- [ ] **Step 4: Run and confirm pass**

Run: `pytest triton_compile/tests/test_einops.py -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add triton_compile/operators.py triton_compile/tests/test_einops.py
git commit -m "feat(triton_compile): Einops operator (rearrange + matmul paths)"
```

---

## Task 9: SoftMax operator

**Files:**
- Modify: `triton_compile/operators.py`
- Create: `triton_compile/tests/test_softmax.py`

Row-wise softmax. Stage A emits a single-pass kernel (load full row into SMEM, compute). This only works for rows that fit in a block; document the limit.

- [ ] **Step 1: Write the failing test**

Create `triton_compile/tests/test_softmax.py`:

```python
from __future__ import annotations

import ast

import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def test_softmax_emits_valid_triton() -> None:
    term = ops.SoftMax.template()
    result = tc_compile(term)
    assert len(result.kernel_sources) == 1
    src = result.kernel_sources[0]
    ast.parse(src)
    assert "tl.exp" in src
    assert "tl.max" in src
    assert "tl.sum" in src
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest triton_compile/tests/test_softmax.py -v`
Expected: FAIL with `NotImplementedError: No TritonOperator registered for SoftMax`.

- [ ] **Step 3: Implement SoftMax**

Append to `triton_compile/operators.py`:

```python
class _SoftMaxTriton(TritonOperator, operation_key=ops.SoftMax):
    def emit(self, target: cat.Broadcasted) -> str:
        body = (
            "\n    row_idx = tl.program_id(0)"
            "\n    BLOCK: tl.constexpr = 1024"
            "\n    col_offsets = tl.arange(0, BLOCK)"
            "\n    mask = col_offsets < n_cols"
            "\n    row_start = row_idx * n_cols"
            "\n    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))"
            "\n    x_max = tl.max(x, axis=0)"
            "\n    numerator = tl.exp(x - x_max)"
            "\n    denominator = tl.sum(numerator, axis=0)"
            "\n    tl.store(y_ptr + row_start + col_offsets, numerator / denominator, mask=mask)"
        )
        return codegen.KernelSource(
            name="_softmax_kernel",
            params=[
                codegen.Param("x_ptr", "pointer"),
                codegen.Param("y_ptr", "pointer"),
                codegen.Param("n_cols", "i32"),
            ],
            body=body,
        ).render()
```

- [ ] **Step 4: Run and confirm pass**

Run: `pytest triton_compile/tests/test_softmax.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add triton_compile/operators.py triton_compile/tests/test_softmax.py
git commit -m "feat(triton_compile): SoftMax operator (single-pass row kernel)"
```

---

## Task 10: WeightedTriangularLower operator

**Files:**
- Modify: `triton_compile/operators.py`
- Create: `triton_compile/tests/test_wtril.py`

Implements `y = tril(x) / (sum(tril(x)) + 1e-8)` per-row — same semantics as `torch_compile/torch_compile.py:260`.

- [ ] **Step 1: Write the failing test**

Create `triton_compile/tests/test_wtril.py`:

```python
from __future__ import annotations

import ast

import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def test_wtril_emits_valid_triton() -> None:
    term = ops.WeightedTriangularLower.template()
    result = tc_compile(term)
    src = result.kernel_sources[0]
    ast.parse(src)
    assert "tl.where" in src
    assert "tl.sum" in src
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest triton_compile/tests/test_wtril.py -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement**

Append to `triton_compile/operators.py`:

```python
class _WTrilTriton(TritonOperator, operation_key=ops.WeightedTriangularLower):
    def emit(self, target: cat.Broadcasted) -> str:
        body = (
            "\n    row_idx = tl.program_id(0)"
            "\n    BLOCK: tl.constexpr = 1024"
            "\n    col_offsets = tl.arange(0, BLOCK)"
            "\n    mask = col_offsets < n_cols"
            "\n    row_start = row_idx * n_cols"
            "\n    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)"
            "\n    tril_mask = col_offsets <= row_idx"
            "\n    x_masked = tl.where(tril_mask & mask, x, 0.0)"
            "\n    denom = tl.sum(x_masked, axis=0) + 1e-8"
            "\n    tl.store(y_ptr + row_start + col_offsets, x_masked / denom, mask=mask)"
        )
        return codegen.KernelSource(
            name="_wtril_kernel",
            params=[
                codegen.Param("x_ptr", "pointer"),
                codegen.Param("y_ptr", "pointer"),
                codegen.Param("n_cols", "i32"),
            ],
            body=body,
        ).render()
```

- [ ] **Step 4: Run and confirm pass**

Run: `pytest triton_compile/tests/test_wtril.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add triton_compile/operators.py triton_compile/tests/test_wtril.py
git commit -m "feat(triton_compile): WeightedTriangularLower operator"
```

---

## Task 11: Normalize operator (LayerNorm)

**Files:**
- Modify: `triton_compile/operators.py`
- Create: `triton_compile/tests/test_normalize.py`

`torch_compile` maps `ops.Normalize` to `nn.LayerNorm`. We mirror that: mean/variance over the last axis + affine scale/shift.

- [ ] **Step 1: Write the failing test**

Create `triton_compile/tests/test_normalize.py`:

```python
from __future__ import annotations

import ast

import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def test_normalize_emits_valid_triton() -> None:
    term = ops.Normalize.template(input_size=64)
    result = tc_compile(term)
    src = result.kernel_sources[0]
    ast.parse(src)
    assert "tl.sum" in src
    assert "tl.rsqrt" in src
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest triton_compile/tests/test_normalize.py -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement**

Append to `triton_compile/operators.py`:

```python
class _NormalizeTriton(TritonOperator, operation_key=ops.Normalize):
    def emit(self, target: cat.Broadcasted) -> str:
        body = (
            "\n    row_idx = tl.program_id(0)"
            "\n    BLOCK: tl.constexpr = 1024"
            "\n    col_offsets = tl.arange(0, BLOCK)"
            "\n    mask = col_offsets < n_cols"
            "\n    row_start = row_idx * n_cols"
            "\n    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)"
            "\n    mean = tl.sum(x, axis=0) / n_cols"
            "\n    x_centered = tl.where(mask, x - mean, 0.0)"
            "\n    var = tl.sum(x_centered * x_centered, axis=0) / n_cols"
            "\n    rstd = tl.rsqrt(var + 1e-5)"
            "\n    gamma = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)"
            "\n    beta = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)"
            "\n    y = x_centered * rstd * gamma + beta"
            "\n    tl.store(y_ptr + row_start + col_offsets, y, mask=mask)"
        )
        return codegen.KernelSource(
            name="_normalize_kernel",
            params=[
                codegen.Param("x_ptr", "pointer"),
                codegen.Param("y_ptr", "pointer"),
                codegen.Param("weight_ptr", "pointer"),
                codegen.Param("bias_ptr", "pointer"),
                codegen.Param("n_cols", "i32"),
            ],
            body=body,
        ).render()
```

- [ ] **Step 4: Run and confirm pass**

Run: `pytest triton_compile/tests/test_normalize.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add triton_compile/operators.py triton_compile/tests/test_normalize.py
git commit -m "feat(triton_compile): Normalize operator (LayerNorm kernel)"
```

---

## Task 12: Linear operator

**Files:**
- Modify: `triton_compile/operators.py`
- Create: `triton_compile/tests/test_linear.py`

Linear = matmul against learned weights + optional bias. We emit a matmul kernel with epilogue-bias.

- [ ] **Step 1: Write the failing test**

Create `triton_compile/tests/test_linear.py`:

```python
from __future__ import annotations

import ast

import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def test_linear_emits_valid_triton() -> None:
    term = ops.Linear.template(input_size=32, output_size=16)
    result = tc_compile(term)
    src = result.kernel_sources[0]
    ast.parse(src)
    assert "tl.dot" in src


def test_linear_with_bias() -> None:
    from dataclasses import replace

    term = ops.Linear.template(input_size=32, output_size=16)
    biased_op = replace(term.operator, bias=True)
    biased = replace(term, operator=biased_op)
    result = tc_compile(biased)
    src = result.kernel_sources[0]
    assert "bias_ptr" in src
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest triton_compile/tests/test_linear.py -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement**

Append to `triton_compile/operators.py`:

```python
class _LinearTriton(TritonOperator, operation_key=ops.Linear):
    def emit(self, target: cat.Broadcasted) -> str:
        has_bias = bool(getattr(target.operator, "bias", False))
        params: list[codegen.Param] = [
            codegen.Param("x_ptr", "pointer"),
            codegen.Param("w_ptr", "pointer"),
        ]
        if has_bias:
            params.append(codegen.Param("bias_ptr", "pointer"))
        params.extend([
            codegen.Param("y_ptr", "pointer"),
            codegen.Param("M", "i32"),
            codegen.Param("N", "i32"),
            codegen.Param("K", "i32"),
        ])
        bias_load = (
            "\n        b = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)"
            "\n        acc += b[None, :]"
        ) if has_bias else ""
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
            "\n        x_ptrs = x_ptr + (offs_m[:, None] * K + (k + offs_k)[None, :])"
            "\n        w_ptrs = w_ptr + ((k + offs_k)[:, None] * N + offs_n[None, :])"
            "\n        x_mask = (offs_m[:, None] < M) & ((k + offs_k)[None, :] < K)"
            "\n        w_mask = ((k + offs_k)[:, None] < K) & (offs_n[None, :] < N)"
            "\n        x = tl.load(x_ptrs, mask=x_mask, other=0.0)"
            "\n        w = tl.load(w_ptrs, mask=w_mask, other=0.0)"
            "\n        acc += tl.dot(x, w)"
            f"{bias_load}"
            "\n    y_ptrs = y_ptr + (offs_m[:, None] * N + offs_n[None, :])"
            "\n    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)"
            "\n    tl.store(y_ptrs, acc, mask=y_mask)"
        )
        return codegen.KernelSource(
            name="_linear_kernel",
            params=params,
            body=body,
        ).render()
```

- [ ] **Step 4: Run and confirm pass**

Run: `pytest triton_compile/tests/test_linear.py -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add triton_compile/operators.py triton_compile/tests/test_linear.py
git commit -m "feat(triton_compile): Linear operator (matmul + optional bias)"
```

---

## Task 13: Embedding operator

**Files:**
- Modify: `triton_compile/operators.py`
- Create: `triton_compile/tests/test_embedding.py`

Gather kernel: read row `x[i]` from embedding table for each token index.

- [ ] **Step 1: Write the failing test**

Create `triton_compile/tests/test_embedding.py`:

```python
from __future__ import annotations

import ast

import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def test_embedding_emits_valid_triton() -> None:
    term = ops.Embedding.template(embedding_size="V", output_size=32)
    result = tc_compile(term)
    src = result.kernel_sources[0]
    ast.parse(src)
    assert "tl.load" in src
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest triton_compile/tests/test_embedding.py -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement**

Append to `triton_compile/operators.py`:

```python
class _EmbeddingTriton(TritonOperator, operation_key=ops.Embedding):
    def emit(self, target: cat.Broadcasted) -> str:
        body = (
            "\n    token_idx = tl.program_id(0)"
            "\n    BLOCK: tl.constexpr = 256"
            "\n    col_offsets = tl.arange(0, BLOCK)"
            "\n    mask = col_offsets < n_cols"
            "\n    idx = tl.load(idx_ptr + token_idx)"
            "\n    emb_row_start = idx * n_cols"
            "\n    emb = tl.load(table_ptr + emb_row_start + col_offsets, mask=mask)"
            "\n    out_row_start = token_idx * n_cols"
            "\n    tl.store(y_ptr + out_row_start + col_offsets, emb, mask=mask)"
        )
        return codegen.KernelSource(
            name="_embedding_kernel",
            params=[
                codegen.Param("idx_ptr", "pointer"),
                codegen.Param("table_ptr", "pointer"),
                codegen.Param("y_ptr", "pointer"),
                codegen.Param("n_cols", "i32"),
            ],
            body=body,
        ).render()
```

- [ ] **Step 4: Run and confirm pass**

Run: `pytest triton_compile/tests/test_embedding.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add triton_compile/operators.py triton_compile/tests/test_embedding.py
git commit -m "feat(triton_compile): Embedding operator (gather kernel)"
```

---

## Task 14: TritonModule runtime (nn.Module wrapper)

**Files:**
- Create: `triton_compile/runtime.py`
- Modify: `triton_compile/triton_compile.py`
- Modify: `triton_compile/__init__.py`
- Create: `triton_compile/tests/test_runtime.py`

`TritonModule` is the `nn.Module` users receive. It owns the compiled kernel functions (the strings compiled to `@triton.jit` callables), parameters, and orchestrates launches by walking the term a second time and dispatching each `Broadcasted` to the right kernel.

Stage A keeps this minimal: `__init__` stores kernel sources as strings only — no `exec`, no `triton` import. Compilation is lazy: the first `forward()` call on a CUDA device exec's each source and caches the resulting `@triton.jit` functions. This means Mac instantiation always works, GPU forward() works, Mac forward() raises.

- [ ] **Step 1: Write the failing test**

Create `triton_compile/tests/test_runtime.py`:

```python
from __future__ import annotations

import pytest

import data_structure.Operators as ops
from triton_compile import TritonModule, compile as tc_compile


def test_compile_returns_triton_module() -> None:
    term = ops.ReLU.template()
    mod = tc_compile(term).to_module()
    assert isinstance(mod, TritonModule)


def test_module_has_kernel_sources() -> None:
    term = ops.ReLU.template()
    mod = tc_compile(term).to_module()
    assert len(mod.kernel_sources) == 1
    assert "@triton.jit" in mod.kernel_sources[0]


@pytest.mark.requires_gpu
def test_module_forward_relu(gpu_available: bool) -> None:
    import torch

    term = ops.ReLU.template()
    mod = tc_compile(term).to_module().cuda()
    x = torch.randn(1024, device="cuda")
    y = mod(x)
    assert torch.allclose(y, torch.relu(x), atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run and confirm failure**

Run: `pytest triton_compile/tests/test_runtime.py -v`
Expected: FAIL — `CompiledTerm` has no `to_module`; `TritonModule` not exported.

- [ ] **Step 3: Implement TritonModule**

Create `triton_compile/runtime.py`:

```python
from __future__ import annotations

import torch
import torch.nn as nn

import data_structure.Category as cat


class TritonModule(nn.Module):
    def __init__(self, term: cat.Morphism, kernel_sources: tuple[str, ...]) -> None:
        super().__init__()
        self.term = term
        self.kernel_sources = kernel_sources
        # Lazy: do not exec sources until forward() on CUDA.
        self._kernels: tuple[object, ...] | None = None

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TritonModule.forward requires a CUDA GPU. "
                "Compilation works on CPU; execution does not."
            )
        if self._kernels is None:
            self._kernels = tuple(_compile_source(src) for src in self.kernel_sources)
        return _run(self.term, self._kernels, xs)


def _compile_source(src: str) -> object:
    import re

    match = re.search(r"^def (_[a-z_]+_kernel)\(", src, re.MULTILINE)
    if not match:
        raise RuntimeError(f"No kernel function name in source:\n{src[:200]}")
    name = match.group(1)
    namespace: dict[str, object] = {}
    exec(src, namespace)  # noqa: S102 - trusted: generated by codegen
    return namespace[name]


def _run(
    term: cat.Morphism,
    kernels: tuple[object, ...],
    xs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    raise NotImplementedError(
        "Stage A runtime: launcher for compiled kernels is deferred to Task 15"
    )
```

- [ ] **Step 4: Add `to_module()` to CompiledTerm**

Modify `triton_compile/triton_compile.py` — replace the `CompiledTerm` dataclass (at the bottom of the file) with:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class CompiledTerm:
    term: cat.Morphism
    kernel_sources: tuple[str, ...]

    def to_module(self) -> "TritonModule":
        from triton_compile.runtime import TritonModule

        return TritonModule(self.term, self.kernel_sources)
```

Update `triton_compile/__init__.py`:

```python
from __future__ import annotations

from triton_compile.runtime import TritonModule
from triton_compile.triton_compile import CompiledTerm, compile

__all__ = ["TritonModule", "CompiledTerm", "compile"]
```

- [ ] **Step 5: Run and confirm pass (first two tests)**

Run: `pytest triton_compile/tests/test_runtime.py -v`
Expected on Mac: first two PASS (no `exec` at construction — sources are stored as strings), third SKIPPED (`no CUDA GPU available`).

- [ ] **Step 6: Commit**

```bash
git add triton_compile/runtime.py triton_compile/triton_compile.py \
        triton_compile/__init__.py triton_compile/tests/test_runtime.py
git commit -m "feat(triton_compile): TritonModule nn.Module wrapper"
```

---

## Task 15: Launcher — connect CompiledTerm to TritonModule.forward

**Files:**
- Modify: `triton_compile/runtime.py`
- Create: `triton_compile/launch.py`
- Create: `triton_compile/tests/test_launch.py`

Walk the term again at forward-time and for each `Broadcasted`, allocate the output, compute the launch grid, call the kernel. Intermediate values are anonymous tensors passed along; parameters (for `Linear`, `Embedding`, `Normalize`) are stored on the module as `nn.Parameter`.

Stage A keeps parameter handling minimal: we reuse `torch_compile.torch_utilities.Multilinear` / `nn.LayerNorm` / `nn.Embedding` to hold weights; forward() extracts tensors from them and passes raw pointers to Triton kernels. This avoids re-inventing parameter initialization.

- [ ] **Step 1: Write the failing test**

Create `triton_compile/tests/test_launch.py`:

```python
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

    term = ops.ReLU.template() @ ops.ReLU.template()
    mod = tc_compile(term).to_module().cuda()
    x = torch.randn(1024, device="cuda")
    y = mod(x)
    assert torch.allclose(y, torch.relu(torch.relu(x)), atol=1e-4)
```

- [ ] **Step 2: Run and confirm failure (on Mac: tests are SKIPPED — that still counts as "does not pass". Proceed.)**

Run: `pytest triton_compile/tests/test_launch.py -v`
Expected on Mac: both SKIPPED. The TDD loop here is effectively GPU-only; the structural tests from earlier tasks cover the non-execution paths.

- [ ] **Step 3: Implement launcher**

Create `triton_compile/launch.py`:

```python
from __future__ import annotations

import math
from typing import Any

import torch

import data_structure.Category as cat
import data_structure.Operators as ops


def launch(
    term: cat.Morphism,
    kernels_by_node_id: dict[int, object],
    inputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    results = _walk(term, kernels_by_node_id, inputs)
    if not isinstance(results, tuple):
        results = (results,)
    return results


def _walk(
    term: cat.Morphism,
    kernels: dict[int, object],
    xs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...] | torch.Tensor:
    match term:
        case cat.Rearrangement():
            return term.apply(xs)
        case cat.ProductOfMorphisms():
            out: list[torch.Tensor] = []
            for sub, (_, sub_xs) in zip(term.content, term.partition(xs)):
                r = _walk(sub, kernels, sub_xs)
                out.extend(r if isinstance(r, tuple) else (r,))
            return tuple(out)
        case cat.Composed():
            current: tuple[torch.Tensor, ...] = xs
            for sub in term.content:
                r = _walk(sub, kernels, current)
                current = r if isinstance(r, tuple) else (r,)
            return current
        case cat.Block():
            return _walk(term.body, kernels, xs)
        case cat.Broadcasted():
            kernel = kernels[id(term)]
            return _dispatch_broadcast(term, kernel, xs)
    raise NotImplementedError(f"Unhandled term type: {type(term).__name__}")


def _dispatch_broadcast(
    term: cat.Broadcasted,
    kernel: Any,
    xs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    op_type = type(term.operator)
    if op_type in (ops.ReLU, ops.Elementwise, ops.Dropout, ops.Identity):
        return _launch_elementwise(kernel, xs[0])
    if op_type is ops.AdditionOp:
        return _launch_addition(kernel, xs[0], xs[1])
    if op_type is ops.SoftMax:
        return _launch_softmax(kernel, xs[0])
    if op_type is ops.WeightedTriangularLower:
        return _launch_wtril(kernel, xs[0])
    raise NotImplementedError(
        f"Stage A launcher: no dispatch for {op_type.__name__}"
    )


def _launch_elementwise(kernel: Any, x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (math.ceil(n / BLOCK),)
    kernel[grid](x, y, n)
    return y


def _launch_addition(kernel: Any, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(a)
    n = a.numel()
    BLOCK = 1024
    grid = (math.ceil(n / BLOCK),)
    kernel[grid](a, b, y, n)
    return y


def _launch_softmax(kernel: Any, x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    kernel[(rows,)](x, y, n_cols)
    return y


def _launch_wtril(kernel: Any, x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    rows = x.shape[-1]
    n_cols = x.shape[-1]
    kernel[(rows,)](x, y, n_cols)
    return y
```

**Note on parameterized operators** (`Linear`, `Embedding`, `Normalize`): these need stored parameters. Stage A defers their launcher glue to Task 16's integration; for now the launcher will raise `NotImplementedError` for them. Task 16 will extend `_dispatch_broadcast` with the param-holding variants when it wires up the transformer.

- [ ] **Step 4: Wire launcher into TritonModule**

Replace `triton_compile/runtime.py` with:

```python
from __future__ import annotations

import re

import torch
import torch.nn as nn

import data_structure.Category as cat
from triton_compile.launch import launch


class TritonModule(nn.Module):
    def __init__(self, term: cat.Morphism, kernel_sources: tuple[str, ...]) -> None:
        super().__init__()
        self.term = term
        self.kernel_sources = kernel_sources
        self._kernels_by_node: dict[int, object] | None = None

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "TritonModule.forward requires a CUDA GPU. "
                "Compilation works on CPU; execution does not."
            )
        if self._kernels_by_node is None:
            self._kernels_by_node = _bind_kernels(self.term, self.kernel_sources)
        result = launch(self.term, self._kernels_by_node, xs)
        return result[0] if len(result) == 1 else result


def _bind_kernels(
    term: cat.Morphism,
    sources: tuple[str, ...],
) -> dict[int, object]:
    broadcasts = _collect_broadcasts(term)
    if len(broadcasts) != len(sources):
        raise RuntimeError(
            f"kernel count {len(sources)} != broadcast count {len(broadcasts)}"
        )
    return {id(b): _compile_source(src) for b, src in zip(broadcasts, sources)}


def _collect_broadcasts(term: cat.Morphism) -> list[cat.Broadcasted]:
    out: list[cat.Broadcasted] = []
    _collect(term, out)
    return out


def _collect(term: cat.Morphism, out: list[cat.Broadcasted]) -> None:
    match term:
        case cat.Rearrangement():
            return
        case cat.ProductOfMorphisms():
            for sub in term.content:
                _collect(sub, out)
        case cat.Composed():
            for sub in term.content:
                _collect(sub, out)
        case cat.Block():
            _collect(term.body, out)
        case cat.Broadcasted():
            out.append(term)


_KERNEL_NAME_RE = re.compile(r"^def (_[a-z_]+_kernel)\(", re.MULTILINE)


def _compile_source(src: str) -> object:
    match = _KERNEL_NAME_RE.search(src)
    if not match:
        raise RuntimeError(f"No kernel function name in source:\n{src[:200]}")
    name = match.group(1)
    namespace: dict[str, object] = {}
    exec(src, namespace)  # noqa: S102 - trusted: source generated by codegen
    return namespace[name]
```

This preserves Task 14's lazy-compilation contract (no `exec` on Mac construction) while wiring in the launcher.

- [ ] **Step 5: Run tests**

Run: `pytest triton_compile/tests/ -v`
Expected on Mac: all structural tests PASS, all `requires_gpu` tests SKIPPED, no unexpected failures.
Expected on GPU: structural tests PASS, `test_launch_single_relu` and `test_launch_composed_relu_relu` PASS. Fix any failures here by inspecting grid/stride math before moving on.

- [ ] **Step 6: Commit**

```bash
git add triton_compile/launch.py triton_compile/runtime.py \
        triton_compile/tests/test_launch.py
git commit -m "feat(triton_compile): kernel launcher, wires compile → forward"
```

---

## Task 16: Integration — README attention example

**Files:**
- Modify: `triton_compile/launch.py`
- Create: `triton_compile/tests/test_integration_attention.py`

Compile the exact attention expression from `README.md`. This exercises `Einops` (both contraction and rearrange paths), `SoftMax`, `WeightedTriangularLower`, and `Composed`. It reveals any gaps in the launcher's dispatch table for matmul and any reshaping issues.

- [ ] **Step 1: Write the failing test**

Create `triton_compile/tests/test_integration_attention.py`:

```python
from __future__ import annotations

import pytest

import construction_helpers.composition  # noqa: F401 - enables `@` composition
import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def _attention_term():
    qk_matmul = ops.Einops.template("q h d, x h d -> h q d")
    softmax = ops.SoftMax.template()
    mask = ops.WeightedTriangularLower.template()
    sv_matmul = ops.Einops.template("h q x, x h d -> q h d")
    return qk_matmul @ softmax @ mask @ sv_matmul


def test_attention_compiles_structurally() -> None:
    term = _attention_term()
    result = tc_compile(term)
    assert len(result.kernel_sources) == 4
    for src in result.kernel_sources:
        import ast
        ast.parse(src)


@pytest.mark.requires_gpu
def test_attention_matches_torch_compile() -> None:
    import torch

    from torch_compile.torch_compile import ConstructedModule

    term = _attention_term()
    triton_mod = tc_compile(term).to_module().cuda()
    torch_mod = ConstructedModule.construct(term).cuda()

    torch.manual_seed(0)
    q = torch.randn(8, 4, 16, device="cuda")
    k = torch.randn(8, 4, 16, device="cuda")
    v = torch.randn(8, 4, 16, device="cuda")

    y_triton = triton_mod(q, k, v)
    y_torch = torch_mod(q, k, v)
    assert torch.allclose(y_triton, y_torch, atol=1e-3, rtol=1e-3)
```

- [ ] **Step 2: Run the structural test**

Run: `pytest triton_compile/tests/test_integration_attention.py::test_attention_compiles_structurally -v`
Expected: PASS. All four operators (Einops×2, SoftMax, WTril) are registered from Tasks 8–10; the structural test only checks `ast.parse` of emitted sources and does not launch.

If it fails, inspect which operator is missing from the registry and fix before moving on — the GPU test in Step 3+ depends on the structural path being green.

- [ ] **Step 3: Extend the launcher with Einops dispatch (for the GPU test)**

Modify `triton_compile/launch.py` — add inside `_dispatch_broadcast`, before the `raise NotImplementedError`:

```python
    if op_type is ops.Einops:
        return _launch_einops(term, kernel, xs)
```

And add the helper at the bottom of the file:

```python
def _launch_einops(
    term: cat.Broadcasted,
    kernel: Any,
    xs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    # Stage A: flatten-and-permute fallback for contraction.
    # Uses torch.einsum to pre-flatten inputs into 2D matmul shape;
    # the Triton kernel handles the dense matmul.
    if len(xs) == 2 and _has_contraction(term.operator):
        a, b = xs
        a2, b2, out_shape = _flatten_for_matmul(term.operator, a, b)
        M, K = a2.shape
        K2, N = b2.shape
        assert K == K2
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        BLOCK_M = BLOCK_N = 64
        grid = (math.ceil(M / BLOCK_M), math.ceil(N / BLOCK_N))
        kernel[grid](a2, b2, c, M, N, K)
        return c.reshape(out_shape)
    # Non-contraction: currently fall back to einops.einsum for the
    # signature, then launch rearrange kernel. For Stage A we accept
    # this hybrid: the Triton rearrange kernel is structurally valid
    # but numerically-equivalent execution goes via einops.
    import einops as einops_pkg

    signature = _signature_str(term.operator)
    return einops_pkg.einsum(*xs, signature)


def _has_contraction(op: ops.Einops) -> bool:
    input_axes: set[int] = set().union(*op.signature[:-1]) if op.signature else set()
    output_axes: set[int] = set(op.signature[-1]) if op.signature else set()
    return bool(input_axes - output_axes)


def _signature_str(op: ops.Einops) -> str:
    # Reuses the DynamicName the operator was templated with.
    return str(op.name) if op.name is not None else ""


def _flatten_for_matmul(
    op: ops.Einops,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    # Placeholder implementation: for signatures beyond simple `m k, k n -> m n`
    # we rely on torch.einsum permutation. Treat this as Stage A fallback;
    # replaced by real weave-aware flattening in Stage B's fusion work.
    import einops as einops_pkg

    target_sig = _signature_str(op)
    y = einops_pkg.einsum(a, b, target_sig)
    return a, b, tuple(y.shape)
```

**Known debt:** this bypasses the Triton matmul kernel for non-`m k, k n` signatures and falls back to `einops.einsum`. Document this in the test file as an expected limitation of Stage A. Stage B's fusion pass will emit a real Triton kernel that subsumes the permutation. Add a TODO comment in the code noting this.

- [ ] **Step 4: Run structural test and confirm pass**

Run: `pytest triton_compile/tests/test_integration_attention.py::test_attention_compiles_structurally -v`
Expected: PASS.

- [ ] **Step 5: Snapshot the emitted Triton source as a golden file**

Create `triton_compile/tests/golden/attention_stageA.triton.py` by running this one-off script (commit only the output, not the script):

```bash
mkdir -p triton_compile/tests/golden
python -c "
import construction_helpers.composition  # enables @ composition
import data_structure.Operators as ops
from triton_compile import compile as tc_compile
qk = ops.Einops.template('q h d, x h d -> h q d')
sm = ops.SoftMax.template()
mask = ops.WeightedTriangularLower.template()
sv = ops.Einops.template('h q x, x h d -> q h d')
term = qk @ sm @ mask @ sv
result = tc_compile(term)
with open('triton_compile/tests/golden/attention_stageA.triton.py', 'w') as f:
    for i, src in enumerate(result.kernel_sources):
        f.write(f'# Kernel {i}\n')
        f.write(src)
        f.write('\n')
"
```

- [ ] **Step 6: Add a golden-file snapshot test**

Append to `triton_compile/tests/test_integration_attention.py`:

```python
def test_attention_matches_golden() -> None:
    from pathlib import Path

    term = _attention_term()
    result = tc_compile(term)
    actual = ""
    for i, src in enumerate(result.kernel_sources):
        actual += f"# Kernel {i}\n{src}\n"
    expected = Path(__file__).parent.joinpath(
        "golden/attention_stageA.triton.py"
    ).read_text()
    assert actual == expected, (
        "Triton source drift. If intentional, regenerate the golden file."
    )
```

Run: `pytest triton_compile/tests/test_integration_attention.py -v`
Expected on Mac: structural + golden PASS, `test_attention_matches_torch_compile` SKIPPED.
Expected on GPU: all three PASS.

- [ ] **Step 7: Commit**

```bash
git add triton_compile/launch.py \
        triton_compile/tests/test_integration_attention.py \
        triton_compile/tests/golden/attention_stageA.triton.py
git commit -m "feat(triton_compile): attention end-to-end (integration + golden)"
```

---

## Task 17: Integration — Transformer block

**Files:**
- Create: `triton_compile/tests/test_integration_transformer.py`
- Create: `triton_compile/tests/golden/transformer_stageA.triton.py`
- Modify: `triton_compile/launch.py` (add Linear/Embedding/Normalize dispatch)

This task closes Stage A exit criterion #2 ("Full Transformer from Transformer.ipynb compiles"). Extends the launcher with the parameter-holding operators.

- [ ] **Step 1: Inspect Transformer.ipynb to extract the model expression**

Run in repo root:

```bash
jupyter nbconvert --to script Transformer.ipynb --stdout 2>/dev/null | head -200
```

Identify the top-level pyncd term defining the transformer block — typically ends with a `@`-chain over `Normalize`, `Linear`, `AdditionOp`, and the attention sub-expression from Task 16. Copy the definition verbatim into the test.

- [ ] **Step 2: Write the failing integration test**

Create `triton_compile/tests/test_integration_transformer.py`:

```python
from __future__ import annotations

import pytest

import construction_helpers.composition  # noqa: F401 - enables `@` composition
import data_structure.Operators as ops
from triton_compile import compile as tc_compile


def _transformer_block():
    # Paste the transformer block expression from Transformer.ipynb here.
    # Placeholder: minimal block = LayerNorm → attention → residual + LayerNorm → MLP.
    attn = (
        ops.Einops.template("q h d, x h d -> h q d")
        @ ops.SoftMax.template()
        @ ops.WeightedTriangularLower.template()
        @ ops.Einops.template("h q x, x h d -> q h d")
    )
    mlp = (
        ops.Linear.template(input_size=64, output_size=256)
        @ ops.ReLU.template()
        @ ops.Linear.template(input_size=256, output_size=64)
    )
    return (
        ops.Normalize.template(input_size=64)
        @ attn
        @ ops.Normalize.template(input_size=64)
        @ mlp
    )


def test_transformer_compiles_structurally() -> None:
    term = _transformer_block()
    result = tc_compile(term)
    for src in result.kernel_sources:
        import ast
        ast.parse(src)


@pytest.mark.requires_gpu
def test_transformer_matches_torch_compile() -> None:
    import torch

    from torch_compile.torch_compile import ConstructedModule

    term = _transformer_block()
    triton_mod = tc_compile(term).to_module().cuda()
    torch_mod = ConstructedModule.construct(term).cuda()

    torch.manual_seed(0)
    x = torch.randn(4, 8, 64, device="cuda")
    y_triton = triton_mod(x)
    y_torch = torch_mod(x)
    assert torch.allclose(y_triton, y_torch, atol=1e-3, rtol=1e-3)
```

- [ ] **Step 3: Run and confirm failure**

Run: `pytest triton_compile/tests/test_integration_transformer.py::test_transformer_compiles_structurally -v`
Expected: FAIL with `NotImplementedError: Stage A launcher: no dispatch for Linear` (or similar).

- [ ] **Step 4: Extend launcher with parameter-holding operators**

Modify `triton_compile/launch.py` — extend `_dispatch_broadcast`:

```python
    if op_type is ops.Linear:
        return _launch_linear(term, kernel, xs)
    if op_type is ops.Embedding:
        return _launch_embedding(term, kernel, xs)
    if op_type is ops.Normalize:
        return _launch_normalize(term, kernel, xs)
```

Add these helpers (parameters are created lazily at first-forward using shapes inferred from the term, mirroring how `torch_compile.ConstructedLinear.__init__` does it):

```python
_PARAM_CACHE: dict[int, dict[str, torch.Tensor]] = {}


def _get_or_init_linear_params(term: cat.Broadcasted) -> dict[str, torch.Tensor]:
    key = id(term)
    if key in _PARAM_CACHE:
        return _PARAM_CACHE[key]
    # Infer (in_size, out_size) from the term's weaves — mirrors
    # torch_compile.ConstructedLinear.__init__.
    in_weave = term.input_weaves[0]
    out_weave = term.output_weaves[0]
    in_size = 1
    for ax in in_weave.target().shape():
        in_size *= ax.local_size()._value  # type: ignore[attr-defined]
    out_size = 1
    for ax in out_weave.target().shape():
        out_size *= ax.local_size()._value  # type: ignore[attr-defined]
    device = "cuda"
    W = torch.empty(in_size, out_size, device=device)
    torch.nn.init.kaiming_uniform_(W, a=5 ** 0.5)
    params = {"W": W}
    if getattr(term.operator, "bias", False):
        params["b"] = torch.zeros(out_size, device=device)
    _PARAM_CACHE[key] = params
    return params


def _launch_linear(
    term: cat.Broadcasted,
    kernel: Any,
    xs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    params = _get_or_init_linear_params(term)
    x = xs[0]
    original_shape = x.shape
    x2 = x.reshape(-1, params["W"].shape[0])
    M, K = x2.shape
    N = params["W"].shape[1]
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    BLOCK_M = BLOCK_N = 64
    grid = (math.ceil(M / BLOCK_M), math.ceil(N / BLOCK_N))
    if "b" in params:
        kernel[grid](x2, params["W"], params["b"], y, M, N, K)
    else:
        kernel[grid](x2, params["W"], y, M, N, K)
    return y.reshape(*original_shape[:-1], N)


def _launch_normalize(
    term: cat.Broadcasted,
    kernel: Any,
    xs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    key = id(term)
    if key not in _PARAM_CACHE:
        n_cols = xs[0].shape[-1]
        _PARAM_CACHE[key] = {
            "weight": torch.ones(n_cols, device=xs[0].device),
            "bias": torch.zeros(n_cols, device=xs[0].device),
        }
    params = _PARAM_CACHE[key]
    x = xs[0]
    y = torch.empty_like(x)
    rows = x.numel() // x.shape[-1]
    n_cols = x.shape[-1]
    kernel[(rows,)](x, y, params["weight"], params["bias"], n_cols)
    return y


def _launch_embedding(
    term: cat.Broadcasted,
    kernel: Any,
    xs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    key = id(term)
    if key not in _PARAM_CACHE:
        V = term.input_weaves[0]._shape  # type: ignore[attr-defined]
        V_size = V if isinstance(V, int) else getattr(V, "_value", None)
        d = 1
        for ax in term.output_weaves[0].target().shape():
            d *= ax.local_size()._value  # type: ignore[attr-defined]
        table = torch.empty(V_size, d, device=xs[0].device)
        torch.nn.init.normal_(table, mean=0.0, std=0.02)
        _PARAM_CACHE[key] = {"table": table}
    params = _PARAM_CACHE[key]
    idx = xs[0]
    rows = idx.numel()
    n_cols = params["table"].shape[1]
    y = torch.empty(rows, n_cols, device=idx.device)
    kernel[(rows,)](idx, params["table"], y, n_cols)
    return y.reshape(*idx.shape, n_cols)
```

**Debt noted:** parameter storage uses a module-level `_PARAM_CACHE` keyed by term `id`. This is Stage A expedient; Stage B will promote parameters to proper `nn.Parameter` on the `TritonModule` for autograd compatibility. Flag this with a TODO at the top of `launch.py`.

- [ ] **Step 5: Run structural test**

Run: `pytest triton_compile/tests/test_integration_transformer.py::test_transformer_compiles_structurally -v`
Expected: PASS.

- [ ] **Step 6: Snapshot golden**

```bash
mkdir -p triton_compile/tests/golden
python -c "
import construction_helpers.composition  # enables @ composition
import data_structure.Operators as ops
from triton_compile import compile as tc_compile
attn = (
    ops.Einops.template('q h d, x h d -> h q d')
    @ ops.SoftMax.template()
    @ ops.WeightedTriangularLower.template()
    @ ops.Einops.template('h q x, x h d -> q h d')
)
mlp = (
    ops.Linear.template(input_size=64, output_size=256)
    @ ops.ReLU.template()
    @ ops.Linear.template(input_size=256, output_size=64)
)
term = (
    ops.Normalize.template(input_size=64)
    @ attn
    @ ops.Normalize.template(input_size=64)
    @ mlp
)
result = tc_compile(term)
with open('triton_compile/tests/golden/transformer_stageA.triton.py', 'w') as f:
    for i, src in enumerate(result.kernel_sources):
        f.write(f'# Kernel {i}\n')
        f.write(src)
        f.write('\n')
"
```

Append golden-file test to `triton_compile/tests/test_integration_transformer.py`:

```python
def test_transformer_matches_golden() -> None:
    from pathlib import Path

    term = _transformer_block()
    result = tc_compile(term)
    actual = ""
    for i, src in enumerate(result.kernel_sources):
        actual += f"# Kernel {i}\n{src}\n"
    expected = Path(__file__).parent.joinpath(
        "golden/transformer_stageA.triton.py"
    ).read_text()
    assert actual == expected, (
        "Transformer Triton source drift. Regenerate golden file if intentional."
    )
```

Run full suite: `pytest triton_compile/tests/ -v`
Expected on Mac: all structural + golden PASS, `requires_gpu` SKIPPED.

- [ ] **Step 7: Commit**

```bash
git add triton_compile/launch.py \
        triton_compile/tests/test_integration_transformer.py \
        triton_compile/tests/golden/transformer_stageA.triton.py
git commit -m "feat(triton_compile): transformer block end-to-end (integration + golden)"
```

---

## Task 18: Update TODO.md

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Append Stage A completion and pointer to Stage B**

Modify `TODO.md` — append:

```markdown
- [x] Stage A: Triton compilation of pyncd BroadcastedCategory terms
  - Unfused: one kernel per Broadcasted
  - Operators: Einops, Elementwise, SoftMax, WTril, Normalize, Linear, Embedding, AdditionOp, Rearrangement
  - See docs/superpowers/plans/2026-04-23-triton-compile-stage-a.md
- [ ] Stage B: Compositional fusion via categorical rewrites
- [ ] Stage C: FAN-derived IO-optimal kernels
```

- [ ] **Step 2: Commit**

```bash
git add TODO.md
git commit -m "docs: mark Triton compile Stage A complete"
```

---

## Stage A — definition of done

- [ ] All 18 tasks committed.
- [ ] `pytest triton_compile/tests/ -v` green on Mac (structural + golden), with `requires_gpu` SKIPPED.
- [ ] On GPU (when RTX 3090 arrives): rerun full suite; all `requires_gpu` tests green, correctness within `atol=1e-3` vs. `torch_compile`.
- [ ] `TODO.md` updated.
- [ ] Human checkpoint before opening Stage B plan.

---

## Known Stage A debt carried into Stage B

Documented inline with `TODO` comments in the code:

1. `launch.py::_launch_einops` falls back to `einops.einsum` for non-trivial signatures; Stage B's fusion pass will emit weave-aware Triton matmul.
2. `launch.py::_PARAM_CACHE` uses module-level dict keyed by `id(term)` — not autograd-safe. Stage B promotes to `nn.Parameter`.
3. `ops.Dropout` currently emits no-op kernel (no RNG state). Deferred until Stage B proves the runtime model.
4. The SoftMax kernel single-passes over a single `BLOCK=1024` block; long sequences not supported in Stage A. Stage B's online softmax via `softmax_contraction_is_streamable` fixes this.
5. No handling for bfloat16 / fp16 in Stage A — all kernels assume fp32. Stage C adds dtype sweeps via the cost model.
