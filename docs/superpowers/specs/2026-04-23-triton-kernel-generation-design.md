# Triton Kernel Generation from pyncd Diagrams — Design

**Date:** 2026-04-23
**Status:** Draft — pending user review
**Target repo:** `pyncd` (Python 3.14)

## 1. Goal

Generate optimized, fused Triton kernels from pyncd's `BroadcastedCategory`
expressions. A user writes a model algebraically (as in `README.md`'s attention
example) and gets back a runnable, fused GPU kernel — with correctness
guarantees relative to `torch_compile` and performance guided by a symbolic
IO-cost model derived from the FlashAttention on a Napkin (FAN) paper.

This is a long-term, iterative project. The design stages the work so each
stage produces a working system and informs the next.

## 2. Non-goals

- Backpropagation. Forward-only; wrap outputs in a PyTorch autograd shim if
  gradients are needed.
- Quantization (FP8, INT4). The pyncd paper's quantized-datatype work is
  separate.
- Data-dependent control flow: MoE dynamic routing, variable-length ragged
  tensors, early exit.
- Distributed / multi-GPU.
- Beating FlashAttention-3 on Hopper. Target is RTX 3090 (Ampere); FA3
  performance parity is a stretch goal for the research phase, not v1.
- Hand-tuned per-architecture kernels. The pipeline is general-by-construction.

## 3. Existing substrate (what we build on)

- `torch_compile/torch_compile.py` — compositional compiler template
  (dispatches on `Operator` type, walks `Composed` / `ProductOfMorphisms` /
  `Block` / `Rearrangement`).
- `data_structure/` — full categorical representation of terms, including
  `Broadcasted`, `Weave`, `StrideCategory`, and `Reindexings`.
- `data_structure_kernels/Kernel.py` — `ChildKernel`, `KernelizedAxis`,
  `Tiling(STREAM|TILE)`, `KernelizedDatatype`, `kernel_functor`. These are
  the type-system primitives Stage C's derivation pipeline populates.
- `graphs/UIDHypergraph.py` — hypergraph form for category-theoretic
  rewriting (pyncd paper §5.1.4).
- `term_utilities/` — `is_mappable_broadcast`, `get_mapping`, etc.
- `construction_helpers/` — autoalignment, Einops signature parsing.

## 4. Architecture

New sibling package `triton_compile/`, mirroring `torch_compile/`:

```
triton_compile/
├── triton_compile.py    # Term → TritonProgram. Walks Composed /
│                         # Product / Block / Broadcasted. Mirrors
│                         # ConstructedModule.construct.
├── operators.py         # Per-operator StreamingInfo + Triton templates.
│                         # Registry pattern mirrors
│                         # ConstructedModule.operation_registry.
├── codegen.py           # Triton source emission: IR builder, block
│                         # pointers, tl.load/tl.store, launch grid.
├── fusion.py            # Fusion planner: groups adjacent Broadcasteds
│                         # into ChildKernels using kernel_functor.
├── rewrites.py          # Hypergraph rewrite rules. Runs on the
│                         # UIDHypergraph form before fusion.
├── cost_model.py        # Hardware hierarchy + symbolic cost model.
├── kernel_seeds.py      # Stage C: proven-streamable primitives from
│                         # FAN Appendix A.3.
├── runtime.py           # TritonModule: nn.Module wrapper holding
│                         # compiled kernels, params, dataflow.
├── hardware/
│   ├── rtx3090.py       # Ampere hierarchy descriptor (primary).
│   ├── h100.py          # Hopper (future).
│   └── a100.py          # Ampere (future).
└── tests/ ...            # see §9
```

**Data flow:**

```
cat.Morphism
  │
  ├── term_utilities + Kernel.kernel_functor (reuse)
  │
  ├── rewrites.apply()           ← hypergraph rewrite pass
  │       (affine composition, Yoneda sliding, Fusion Theorem)
  │
  ├── fusion.plan()              ← groups Broadcasteds into ChildKernels
  │       calls cost_model.solve() for tile sizes
  │
  ├── codegen.emit()             ← walks fused plan; emits Triton source
  │       calls operators.StreamingInfo.triton_template per op
  │
  └── runtime.TritonModule       ← @triton.jit kernels + dataflow
```

Stage A lights up `triton_compile.py`, `operators.py`, `codegen.py`, and
`runtime.py`. Stage B adds `rewrites.py` and `fusion.py`. Stage C populates
`kernel_seeds.py` and `cost_model.py` fully and extends the rewrite library.

## 5. Staged scope

### 5.1 Stage A — Unfused Triton (Mac phase)

One kernel per `Broadcasted`. No cross-op fusion. `Rearrangement`s compile
into address math (free). `Composed` chains become sequential kernel launches.

**Operators covered (same set as `torch_compile`):**
`Einops`, `Elementwise` / `ReLU` / `Dropout`, `SoftMax`, `Linear`,
`Embedding`, `Normalize`, `AdditionOp`, `WeightedTriangularLower`,
`Rearrangement`.

**Exit criteria:**
1. The README attention example (`qk_matmul @ softmax @ mask @ sv_matmul`)
   compiles to a valid `TritonModule`.
2. The full Transformer from `Transformer.ipynb` compiles.
3. Structural tests pass (§9 Layer 1).
4. Per-op numeric equivalence: `triton_compile ≈ torch_compile` via
   `triton.runtime.interpreter` on toy inputs (atol/rtol = 1e-4). On GPU:
   same check on real hardware.

### 5.2 Stage B — Compositional fusion (Mac codegen, GPU verify)

Fuse adjacent `Broadcasted`s sharing a tiled axis. Fusion is driven by the
rewrite rules of §7, not by hand-written per-pattern matchers.

**Exit criteria:**
1. README attention compiles to **one** fused kernel on GPU.
2. `RMSNorm @ Linear` compiles to one fused kernel.
3. Attention perf on RTX 3090 within 2× of `torch.compile`'s best.
   (Real target set after first measurement.)
4. Correctness unchanged from Stage A.

### 5.3 Stage C — FAN-derived IO-optimal kernels (Mac derive, GPU verify)

Implements the FAN §5 eight-step derivation pipeline as composable passes,
backed by streamable-kernel seeds (FAN App. A.3) and a symbolic multi-level
cost model (FAN §4).

**What "IO-optimal" means here:** for each fused kernel, a configuration
`g⃗*` minimizing `H* = Σ_ℓ Ḣ_ℓ⁻¹ · H*(a⃗, M_ℓ)` (FAN eq. 2) under per-level
memory constraints, derived symbolically — no search.

**Exit criteria:**
1. Matmul derives the FAN §3.1 config table (`g_a = g_c = √M`).
2. Attention derives the FAN §3.2 config table (`g_q ≤ M/2d`).
3. Attention + RMSNorm + Linear on RTX 3090 at ≥ `torch.compile` on at
   least one sequence-length bucket.
4. `calibration.json` populated with measured-vs-predicted deltas per
   kernel; gap < 15% or diagnosed to a specific FAN-listed assumption.

### 5.4 Stage D — research extensions (not v1)

Register-fragmented tensor-core ops, reindexing-into-fragmented-memory
(CUTLASS layouts), cross-core reduction accumulators, wave quantization,
backprop via Para. Tracked but out of scope.

## 6. Per-operator streaming metadata

In `triton_compile/operators.py`, each pyncd `Operator` subclass gets a
declaration:

```python
@dataclass(frozen=True)
class StreamingInfo:
    kind: OpKind                    # POINTWISE | REDUCTION | MATMUL |
                                    # GATHER | REINDEX_ONLY |
                                    # FUSED_SOFTMAX_REDUCTION
    streamable_axes: frozenset[AxisRole]
    accumulator: Callable | None    # FAN Fig. 38 B function
    fuses_as_prologue: bool
    fuses_as_epilogue: bool
    triton_template: Callable[[FusedContext], str]
    cost_coefficients: Callable[[Shapes, Tiles], dict[str, sympy.Expr]]
```

**Seed table (v1):**

| Operator                        | kind                        | streamable_axes     | fuses as            |
| ------------------------------- | --------------------------- | ------------------- | ------------------- |
| `Elementwise` / `ReLU` / `Dropout` | POINTWISE                | all                 | prologue + epilogue |
| `AdditionOp`                    | POINTWISE                   | all                 | prologue + epilogue |
| `Einops` (contraction)          | MATMUL                      | contracted axes     | epilogue only       |
| `Einops` (no contraction)       | POINTWISE                   | all                 | prologue + epilogue |
| `SoftMax`                       | REDUCTION                   | —                   | epilogue only       |
| `SoftMax @ Einops(contraction)` | FUSED_SOFTMAX_REDUCTION     | contracted axis     | epilogue only       |
| `Normalize` (RMS / Layer)       | REDUCTION (Welford)         | —                   | epilogue only       |
| `Embedding`                     | GATHER                      | none                | epilogue only       |
| `Linear`                        | MATMUL                      | input axis          | epilogue only       |
| `WeightedTriangularLower`       | POINTWISE-masking           | all                 | prologue + epilogue |
| `Rearrangement`                 | REINDEX_ONLY                | all                 | free — no kernel    |

## 7. Rewrite rules

Rewrites in `triton_compile/rewrites.py` operate on `UIDHypergraph`.

```python
@dataclass(frozen=True)
class RewriteRule:
    name: str
    pattern: HypergraphPattern
    guard: Callable[[Match], bool]
    rewrite: Callable[[Match], Hypergraph]
    provenance: str
```

**Seed rules (v1):**

1. **`affine_composition`** — `[η₁] ∘ [η₂] → [η₁ ∘ η₂]`.
   *Provenance:* St is a category (pyncd §4, Def. 8).
2. **`reindexing_through_pointwise`** — slide `[η]` through any POINTWISE op.
   *Provenance:* Yoneda naturality (pyncd §4.3, Fig. 19).
3. **`epilogue_absorb_pointwise`** — pull POINTWISE successor into
   predecessor's kernel epilogue. *Provenance:* standard.
4. **`prologue_absorb_pointwise`** — symmetric.
5. **`fusion_theorem`** — streamable F composed with E-weaved-by-streamable-
   axis preserves streamability, yields new accumulator B'.
   *Provenance:* FAN Thm 1, App. A.1.
6. **`weave_preserves_streamability`** — weaving F over a non-streamable axis
   yields an algorithm streamable along the original axis.
   *Provenance:* FAN Fig. 40.
7. **`softmax_contraction_is_streamable`** — recognise
   `SoftMax @ Einops(contraction)`; emit `FUSED_SOFTMAX_REDUCTION` with
   the online-softmax accumulator. *Provenance:* FAN App. A.3.2.
8. **`matmul_is_streamable`** — recognise dot-product contraction; emit
   streaming-K accumulator. *Provenance:* FAN App. A.3.1.
9. **`split_k_split`** — matmul along `k` decomposes as
   `k = k_outer × k_inner` and accumulates linearly.
   *Provenance:* FAN §5.2, Fig. 30.

Rule application runs to fixpoint. Hypergraph form makes matches
order-agnostic.

## 8. Streamable kernel seeds + cost model

### 8.1 Streamable kernel seeds (`triton_compile/kernel_seeds.py`)

Each seed is a primitive with `(operation, accumulator B, head/tail)` from
FAN App. A.3:

| Seed                  | Operation              | Accumulator                    | Source       |
| --------------------- | ---------------------- | ------------------------------ | ------------ |
| `dot_product`         | `Σ x[i] · y[i]`        | `acc + x · y`                  | FAN A.3.1    |
| `softmax_contraction` | `softmax(s) · v`       | online-softmax `(m', l', o')` | FAN A.3.2    |
| `rms_norm`            | `x / √(Σ x² / n)`      | Welford (folded)               | standard     |
| `layer_norm`          | `(x − μ) / σ · γ + β`  | Welford                        | standard     |

Flash-attention derives as `softmax_contraction` + `fusion_theorem` +
`weave(q, d)` + `split_k_split` — applied mechanically by the rewrite engine.

### 8.2 Cost model (`triton_compile/cost_model.py`)

```python
@dataclass(frozen=True)
class Level:
    name: str                 # "GMEM" | "L2" | "SMEM" | "REG" | "TC"
    M_max: int                # bytes
    H_inverse: float          # inverse bandwidth, sec/byte
    N_max: int                # children per parent (for cross-transfer)
    allowed_ops: frozenset[OpKind]
    divisor_constraints: dict[str, int]   # e.g. coalesce=128

@dataclass(frozen=True)
class Hierarchy:
    levels: tuple[Level, ...]  # top-to-bottom

    def total_cost(
        self,
        cost_coeffs: dict[str, sympy.Expr],
        tiles_per_level: dict[str, Shape],
    ) -> sympy.Expr:
        # FAN eq. 2:  H* = Σ_ℓ Ḣ_ℓ⁻¹ · H*(a⃗, M_ℓ)
        ...

    def solve(
        self,
        cost_coeffs: dict[str, sympy.Expr],
        axis_sizes: Shape,
    ) -> TileAssignment:
        # Symbolic minimization via sympy on the Lagrangian.
        # Sum-of-monomial objectives (FAN §3, §4.1) have closed
        # forms — no search required for patterns FAN covers
        # (matmul, attention, grouped query attention). For
        # patterns without a closed form, falls back to numeric
        # search over a bounded tile-size grid.
        ...
```

One `Hierarchy` instance per GPU; RTX 3090 (Ampere) is day-one, seeded with:
- GMEM (24 GB, 936 GB/s)
- L2 (6 MB)
- SMEM (128 KB / SM, 82 SMs)
- Registers (256 KB / SM)
- Tensor cores (Ampere; divisibility: 16×16 for FP16)

**Falsifiability hook:** `Level` constants overridable by `calibration.json`,
written by Loop 4 (§10). When measured perf deviates from predicted, the
calibrator records which assumption moved — FAN §6's falsifiability baked in.

## 9. Test harness

```
triton_compile/tests/
├── conftest.py          # @requires_gpu, golden-file helpers
├── test_codegen.py      # Layer 1: structural (Mac)
├── test_correctness.py  # Layer 2: numeric (Mac interp, GPU real)
├── test_rewrites.py     # Layer 3: rewrite semantic equivalence (Mac)
├── test_costmodel.py    # Layer 4: symbolic cost model (Mac)
├── test_benchmarks.py   # GPU-only, @requires_gpu
└── golden/              # Frozen Triton source per (model, stage)
```

**Layer 1 — structural (Mac, fast):**
- `ast.parse` round-trip on emitted source.
- Launch grid computable; tile sizes within `Hierarchy.M_max`.
- Golden-file snapshot of canonical expressions. Catches silent regressions.

**Layer 2 — correctness:**
- Each op + canonical model has a `test_correctness::test_{op}` case.
- `triton_compile(term)(inputs) ≈ torch_compile(term)(inputs)` with matched
  random seeds.
- Mac: `triton.runtime.interpreter` where supported; `xfail(reason="no-gpu")`
  otherwise.
- GPU: full coverage, sweep `{fp32, fp16, bf16}`.

**Layer 3 — rewrite correctness (Mac):**
- Each rewrite rule applied to a fixture; both pre- and post-rewrite forms
  compiled through `torch_compile` and diffed on common inputs. Catches
  algebraically unsound rules.
- `softmax_contraction_is_streamable`: assert bit-level equality (within
  fp32 epsilon) between naive and online form. Highest-risk rule.

**Layer 4 — cost model (Mac):**
- `cost_model.solve(...)` returns FAN's closed forms (matmul √M, attention
  M/2d).
- When GPU available: `test_costmodel_calibration` takes measurements,
  writes `calibration.json`; subsequent runs assert
  `measured ≤ predicted · (1 + tolerance)`.

**Benchmarks** (GPU-only, excluded from CI): `test_benchmarks.py` vs.
`torch.compile` baseline. Emits a Markdown report per run.

## 10. Autoresearch loop configurations

Four scheduled loops. Each has pre-conditions, a tight success signal,
bounded step budget, and a real exit criterion. No open-ended loops.

### Loop 1 — Stage-A operator coverage (Mac, day 1)
- **Job:** for each `Operator` lacking a `StreamingInfo` entry, draft one
  + Triton template + codegen test + correctness test. Commit on green.
- **Signal:** Layer-1 + Layer-2 green.
- **Budget:** 1 operator per invocation; max 3 invocations before human
  checkpoint on consecutive failures.
- **Exit:** §6 seed table fully covered.
- **Schedule:** hourly until exit.

### Loop 2 — Stage-B rewrite rule expansion (Mac, after Loop 1)
- **Job:** pick one unimplemented rule from §7; derive preconditions;
  implement; add Layer-3 test with fixture; verify it fires on a canonical
  pattern (attention / RMSNorm+Linear / conv).
- **Signal:** Layer-3 green + rule fires on ≥ 1 canonical fixture.
- **Budget:** 1 rule per invocation.
- **Exit:** Stage B exit criteria met.

### Loop 3 — Stage-C derivation pipeline (Mac derive, GPU verify)
- **Job:** pick one canonical model (matmul → attention → transformer
  block); run FAN §5 eight-step derivation; diff derived config table
  against the paper's; fix derivation code or document the diverging
  assumption.
- **Signal:** derived config matches FAN's published table, or diff is
  localised to a named assumption.
- **Budget:** 1 model per invocation; max 5 iterations per model.
- **Exit:** matmul + attention both derive their FAN tables.

### Loop 4 — GPU calibration (GPU-only, post-3090)
- **Job:** run benchmarks, diff measured vs. predicted, update
  `calibration.json`, log which assumption moved.
- **Signal:** `|measured − predicted| / predicted` decreases or plateaus
  with a diagnosed gap.
- **Budget:** 10 runs per invocation.
- **Exit:** < 15% gap on attention + matmul, or gap pinned to a specific
  FAN-listed assumption (e.g., small-tile tensor-core overhead).
- **Schedule:** daily.

**Loops do NOT:** pick priority order (they consume an ordered backlog);
make API or architecture changes; edit outside `triton_compile/` or
`calibration.json` (path-filter enforced).

**Human checkpoints:** every stage exit, every loop exceeding its failure
budget, any edit proposed outside the allowed paths.

**Implementation:** `superpowers:schedule` (cron) for Loop 4; `superpowers:loop`
(dynamic) for Loops 1–3. Exit criteria embedded in each loop's prompt so it
terminates.

## 11. Inherited limits (from FAN §5.8 and §6)

- No tensor-core overhead model for small tiles.
- Assumes latency is fully hidden.
- No register-level tensor-core fragment manipulation.
- No reindexing-into-fragmented-memory (CUTLASS-style layouts).
- No cross-core reduction accumulators.
- No wave quantization.

Stage C is *a sketch of an optimal algorithm*, not a guaranteed SOTA kernel.
Target fidelity: comparable to FAN's prediction of FlashAttention-3's
60–75% utilization (FAN §6.1–6.2).

## 12. Dependencies

- `triton` (pypi) — codegen target.
- `sympy` (pypi) — symbolic cost model.
- existing pyncd deps: `torch`, `einops`, Python 3.14.

## 13. Planning scope

This spec covers Stages A, B, and C plus the loop infrastructure. The
implementation plan generated from this spec should cover **Stage A only**
— a separate plan is written at each stage boundary, informed by what was
learned in the previous stage. This keeps plans sized to one execution
session and avoids locking in Stage B/C decisions before Stage A's reality
is in.

## 14. Success at the end of v1

- README attention example runs on RTX 3090 as a single fused kernel,
  numerically equivalent to `torch_compile`, at ≥ `torch.compile` perf on
  at least one configuration.
- Any model writable in pyncd's category compiles (possibly unfused).
- `calibration.json` with measured-vs-predicted deltas for matmul,
  attention, RMSNorm+Linear.
- Test suite runs on Mac with only Layer-2/benchmark steps gated on GPU.
- Four autoresearch loops configured with terminating exit criteria.
