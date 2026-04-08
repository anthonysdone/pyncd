from __future__ import annotations
from typing import Self, Iterable
from dataclasses import dataclass, field
from enum import Enum
import data_structure.Numeric as nm
import data_structure.Term as fd
import data_structure.Category as cat

type Kernel = ChildKernel | SpecialKernel

class SpecialKernel(Enum):
    MAIN = 'MAIN'
fd.register_enum(SpecialKernel)

@dataclass(frozen=True)
class ChildKernel(fd.UTerm):
    parent: Kernel = SpecialKernel.MAIN

    def lineage(self: Self | SpecialKernel) -> fd.Prod[Kernel]:
        if self == SpecialKernel.MAIN:
            return (SpecialKernel.MAIN,)
        return (self, *ChildKernel.lineage(self.parent))
    

''' Kernelized Stride '''

class Strategy(Enum):
    STREAM = 'STREAM'
    TILE = 'TILE'
fd.register_enum(Strategy)

@dataclass(frozen=True)
class KernelizedAxis(cat.Axis):
    parent: cat.Axis = field(default_factory=lambda: cat.RawAxis())
    kernel: ChildKernel = field(default_factory=lambda: ChildKernel())

@dataclass(frozen=True)
class Tiling(KernelizedAxis):
    strategy: Strategy = Strategy.TILE
    tile_numer: nm.FreeNumeric = nm.FreeNumeric.field()

type KStrideCategory = cat.StrideCategory[cat.RawAxis | KernelizedAxis]

''' Kernelized Broadcasted '''
@dataclass(frozen=True)
class KernelizedDatatype[B:cat.Datatype=cat.Reals](cat.Datatype): # type: ignore
    form: B = cat.Reals() # type: ignore
    kernel: ChildKernel = field(default_factory=lambda: ChildKernel())

type KAxis = cat.RawAxis | KernelizedAxis
type KDatatype[B:cat.Datatype] = B | KernelizedDatatype[B]
type KArray[B:cat.Datatype] = cat.Array[KDatatype[B], KAxis]
type KWeave[B:cat.Datatype] = cat.Weave[KDatatype[B], KAxis]
type KBroadcasted[B:cat.Datatype] = cat.Broadcasted[KDatatype[B], KAxis]
type KBroadcastedCategory[B:cat.Datatype] = cat.BroadcastedCategory[KDatatype[B], KAxis]

''' Kernel Utilities '''
def kernel_functor[S:fd.Term](kernel: Kernel, target: S) -> S:
    match target:
        case ((ChildKernel() as k) | KernelizedAxis(kernel=k)):
            if k in ChildKernel.lineage(kernel):
                return target
            return kernel_functor(kernel, target.parent) # type: ignore
        case KernelizedDatatype(form=form, kernel=k):
            meet_kernel = kernel_functor(kernel, k)
            if meet_kernel == SpecialKernel.MAIN:
                return form
            return target.reconstruct(kernel=meet_kernel)
    return fd.deep_reconstruct(target, lambda t: kernel_functor(kernel, t))
    