from __future__ import annotations

# TODO(Stage B): _PARAM_CACHE uses module-level dict keyed by id(term).
# Not autograd-safe. Stage B promotes weights to nn.Parameter on TritonModule.
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
    if op_type is ops.Einops:
        return _launch_einops(term, kernel, xs)
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
    x_flat = x.reshape(-1, x.shape[-1])
    y_flat = torch.empty_like(x_flat)
    rows, n_cols = x_flat.shape
    kernel[(rows,)](x_flat, y_flat, n_cols)
    return y_flat.reshape(x.shape)


def _launch_wtril(kernel: Any, x: torch.Tensor) -> torch.Tensor:
    x_flat = x.reshape(-1, x.shape[-1])
    y_flat = torch.empty_like(x_flat)
    rows, n_cols = x_flat.shape
    kernel[(rows,)](x_flat, y_flat, n_cols)
    return y_flat.reshape(x.shape)


def _launch_einops(
    term: cat.Broadcasted,
    kernel: Any,
    xs: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    # Stage A: flatten-and-permute fallback for contraction.
    # TODO(Stage B): replace with weave-aware Triton matmul for non-trivial signatures.
    if len(xs) == 2 and _has_contraction_einops(term.operator):
        a, b = xs
        import einops as einops_pkg
        target_sig = _einops_signature_str(term.operator)
        y = einops_pkg.einsum(a, b, target_sig)
        return y
    # Non-contraction: fall back to einops.einsum for permutation.
    import einops as einops_pkg
    target_sig = _einops_signature_str(term.operator)
    return einops_pkg.einsum(*xs, target_sig)


def _has_contraction_einops(op: ops.Einops) -> bool:
    return len(op.signature) > 1


def _einops_signature_str(op: ops.Einops) -> str:
    return op.name.body if op.name is not None else ""
