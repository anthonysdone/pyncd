from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Any,
    Type,
    Callable,
)
import math
from abc import ABC

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import term_utilities.term_utilities as tutil
import data_structure.Category as cat
import data_structure.Operators as ops

import data_structure.BrTyping as Br

import torch
import torch.nn as nn
import einops

import torch_compile.bcast as bcast
import torch_compile.torch_utilities as torch_utilities

@dataclass(frozen=True)
class TorchFunctionInfo:
    explicit_dim: bool = False
    semantic: bool = False
    implicit_lower: bool = True

class ConstructedModule[M: cat.Morphism](nn.Module, ABC):
    operation_registry: dict[Type[cat.Operator], Type[ConstructedModule]] = {}
    functions_registry: dict[Type[cat.Operator], tuple[Callable, TorchFunctionInfo]] = {}

    def __init_subclass__(cls, operation_key: Type[cat.Operator] | None = None):
        super().__init_subclass__()
        if operation_key:
            ConstructedModule.operation_registry[operation_key] = cls

    @classmethod
    def construct(cls, target: cat.Morphism, dim: None | int = None) -> ConstructedModule | Lambda:
        match target:
            case cat.Rearrangement():
                return ConstructedRearrangement(target)
            case cat.ProductOfMorphisms():
                return ConstructedProduct(target)
            case cat.Composed():
                return ConstructedComposed(target)
            case cat.Block():
                return ConstructedBlock(target)
            case cat.Broadcasted():
                return ConstructedModule.construct_broadcasted(target)
        print(target)
        raise NotImplementedError()
    
    @classmethod
    def construct_broadcasted(cls, target: cat.Broadcasted) -> ConstructedModule | Lambda:
        operator_type = type(target.operator)
        if operator_type in cls.operation_registry:
            return cls.operation_registry[operator_type](target)
        elif operator_type in cls.functions_registry:
            return Lambda(target)
        else:
            raise NotImplementedError(f'No constructor found for operator type {operator_type}')

    @classmethod
    def add_function(cls, 
                     func_type: Type[cat.Operator], 
                     func: Callable, 
                     dim: bool = False,
                     semantic: bool = False):
        ConstructedModule.functions_registry[func_type] = (
            func,
            TorchFunctionInfo(
                explicit_dim = dim,
                semantic = semantic
            )
        )

    def __init__(self, target: M, func_info: TorchFunctionInfo = TorchFunctionInfo()) -> None:
        super().__init__()
        self.target = target
        self.func_info = func_info
    

class Lambda(nn.Module):
    def __init__(self, target: cat.Broadcasted):
        super().__init__()
        self.target = target
        self.func, self.func_info = ConstructedModule.functions_registry[
            type(target.operator)
        ]
        self.func = broadcast_func(target, self.func, self.func_info)

    def forward(self, *xs: torch.Tensor):
        return self.func(*xs)
    
    def __repr__(self):
        return f'{type(self).__qualname__}({self.target.operator})'


##################
#### CATEGORY ####
##################

class ConstructedBlock[B: cat.Datatype, A: cat.Axis](
    ConstructedModule[Br.Block[B, A]]
):
    def __init__(self, target: Br.Block[B, A]):
        super().__init__(target)
        if isinstance(repetition := target.block_tag.repetition, nm.Integer) and repetition._value > 1:
            self.module = nn.Sequential(
                *(ConstructedModule.construct(target.body) for _ in range(repetition._value))
            )
        else:
            self.module = ConstructedModule.construct(target.body)

    def forward(self, *xs: torch.Tensor):
        if isinstance(self.module, nn.Sequential):
            for module in self.module:
                xs = to_tuple(module(*xs))
            return xs
        else:
            return self.module(*xs)
    
class ConstructedProduct[B: cat.Datatype, A: cat.Axis](
    ConstructedModule[Br.ProductOfMorphisms[B, A]]
    ):
    def __init__(self, target: Br.ProductOfMorphisms[B, A]):
        super().__init__(target)
        fs = map(ConstructedModule.construct, target.content)
        self.content = nn.ModuleList(fs)
        
    def forward(self, *xs: torch.Tensor) -> fd.Prod[torch.Tensor]:
        return tuple(
            y
            for f, (_, x)
            in zip(self.content, self.target.partition(xs))
            for y in to_tuple(f(*x))
        )
    
def to_tuple[T](x: T | tuple[T, ...]) -> tuple[T, ...]:
    return x if isinstance(x, tuple) else (x,)

class ConstructedComposed[B: cat.Datatype, A: cat.Axis](
    ConstructedModule[Br.Composed[B, A]]
):
    def __init__(self, target: Br.Composed[B, A]):
        super().__init__(target)
        fs = map(ConstructedModule.construct, target.content)
        self.chain = nn.Sequential(*fs)

    def forward(self, *xs):
        for module in self.chain:
            xs = to_tuple(module(*xs))
        return xs
    
class ConstructedRearrangement(ConstructedModule):
    def __init__(self, target: cat.Rearrangement):
        super().__init__(target)

    def forward(self, *xs: torch.Tensor):
        return self.target.apply(xs)

##################
## BROADCASTING ##
##################

def broadcast_func(
        target: cat.Broadcasted, 
        func: Callable,
        broadcast_info: TorchFunctionInfo = TorchFunctionInfo()):
    
    assert tutil.is_mappable_broadcast(target)
    # For now, we just use Einops
    displacement = bcast.get_displacement(target)
    match broadcast_info:
        case TorchFunctionInfo(explicit_dim=True) if displacement is not None:
            def dim_func(*xs: torch.Tensor, **kwargs: Any):
                return func(*xs, **kwargs, dim=displacement)
            return dim_func
        case TorchFunctionInfo(implicit_lower=True) if displacement == -1:
            return func
        case TorchFunctionInfo(semantic=True) if bcast.is_semantically_broadcastable(target):
            degree_size = len(target.degree())
            if all(tutil.is_identity(eta) for eta in target.reindexings):
                return func
            def _func(*xs: torch.Tensor):
                xs = tuple(
                    x.reshape(bcast.unsqueeze_guide(
                        degree_size,
                        tutil.get_mapping(eta),
                        x.shape
                    ))
                    if not tutil.is_identity(eta) else x
                    for x, eta, weave in zip(xs, target.reindexings, target.input_weaves)
                )
                return func(*xs)
            return _func
        case _ if bcast.vmappable(target):
            vmap_guide = bcast.broadcast_vmap(target)
            for input_loc, output_loc in vmap_guide:
                func = torch.vmap(func, in_dims=input_loc, out_dims=output_loc)
            return func
        case _:
            raise NotImplementedError()

################
## OPERATIONS ##
################

############
## STATIC ##
############

def generate_einops_signature(target: cat.Broadcasted[Any, Any, ops.Einops]):
    assert tutil.is_mappable_broadcast(target)
    operator = target.operator
    degree_tags = tuple(
        f'y{i}' for i, _ in enumerate(target.degree()))
    input_tags = tuple(
        f'x{i}' for i in set[int]().union(*target.operator.signature)
    )
    input_segments = (
        input_weave.imprint_axes(
            (degree_tags[i] for i in tutil.get_mapping(eta)),
            (input_tags[j] for j in signature_segment),
        )
        for input_weave, signature_segment, eta in 
        zip(
            target.input_weaves, 
            operator.signature,
            target.reindexings
        )
    )
    input_signature = ', '.join(
        '... ' + ' '.join(segment)
        for segment in input_segments
    )
    output_signature = ' '.join(degree_tags)
    return f'{input_signature} -> ... {output_signature}'

class ConstructedEinops[B: cat.Datatype, A: cat.Axis](
    ConstructedModule[cat.Broadcasted[B, A, ops.Einops]],
    operation_key=ops.Einops
):
    def __init__(self, target: cat.Broadcasted[B, A, ops.Einops]):
        super().__init__(target)
        self.signature = generate_einops_signature(self.target)

    def forward(self, *xs: torch.Tensor):
        return einops.einsum(*xs, self.signature) # type: ignore
    

ConstructedModule.add_function(ops.SoftMax, torch.softmax, dim=True)
ConstructedModule.add_function(ops.AdditionOp, lambda x, y: x + y, semantic=True)
ConstructedModule.add_function(ops.Elementwise, torch.relu)

def weighted_triangular_lower(x: torch.Tensor) -> torch.Tensor:
    trilled = torch.tril(x)
    return trilled / (torch.sum(trilled, dim=-1, keepdim=True) + 1e-8)

ConstructedModule.add_function(ops.WeightedTriangularLower, weighted_triangular_lower)

#############
## LEARNED ##
#############
class ConstructedLinear[
    B: cat.Datatype, A: cat.Axis
](ConstructedModule, operation_key=ops.Linear):
    def __init__(self, target: cat.Broadcasted[B, A, ops.Linear]):
        super().__init__(target)
        match target.input_weaves, target.output_weaves:
            case [
                [cat.Weave() as weave_in],
                [cat.Weave() as weave_out]
            ]:
                self.in_size: fd.Prod[int] = tuple(x.local_size()._value for x in weave_in.target().shape()) # type: ignore
                self.out_size: fd.Prod[int] = tuple(y.local_size()._value for y in weave_out.target().shape()) # type: ignore
            case _:
                assert False
        self.bias = target.operator.bias
        self.module = torch_utilities.Multilinear(
            self.in_size, self.out_size, self.bias
        )
        self.func = broadcast_func(
            target, 
            self.module.forward)

    def forward(self, *xs):
        return self.func(*xs)
    
class ConstructedEmbedding[
    A: cat.Axis
](ConstructedModule, operation_key=ops.Embedding):
    def __init__(self, target: cat.Broadcasted[cat.Datatype, A, ops.Embedding]):
        super().__init__(target)
        match target.input_weaves, target.output_weaves:
            case [
                [cat.Weave(cat.Natural(size)) as weave_in],
                [cat.Weave() as weave_out]
            ]:
                self.num_embeddings = size._value # type: ignore
                self.dims = tuple(y.local_size()._value for y in weave_out.target().shape()) # type: ignore
            case _:
                assert False
            
        self.module = torch.nn.Embedding(
            self.num_embeddings, math.prod(self.dims)
        )

        def func(*xs):
            x = xs[0]
            original_shape = x.shape
            x = x.view(-1)
            embedded = self.module(x)
            return embedded.view(*original_shape, *self.dims)
        
        self.func = broadcast_func(target, func)
    
    def forward(self, *xs):
        return self.func(*xs)
    
class ConstructedNorm[B: cat.Datatype, A: cat.Axis](ConstructedModule, operation_key=ops.Normalize):
    def __init__(self, target: cat.Broadcasted[B, A, ops.Normalize]):
        super().__init__(target)
        self.module = nn.LayerNorm(target.input_weaves[0]._shape[-1].local_size()._value) # type: ignore
        self.func = broadcast_func(target, self.module.forward)

    def forward(self, *xs: torch.Tensor):
        return self.func(xs[0])
