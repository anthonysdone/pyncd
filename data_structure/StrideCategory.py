from __future__ import annotations
from dataclasses import dataclass, field
from typing import (
    Any,
    Self,
    Type,
    TypeVar,
    Callable,
    Iterable,
    overload,
    Sequence,
    Iterator,
)
import random
import math
from abc import ABC
from enum import Enum

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import utilities.utilities as util
import data_structure.ProductCategory as pc

type StrideCategory[A:Axis] = pc.ProdCategory[A, StrideMorphism[A]]

@dataclass(frozen=True)
class Axis(fd.UTerm, ABC):
    _size: nm.Numeric = nm.FreeNumeric.field()
    def local_size(self) -> nm.Numeric:
        return self._size
    @classmethod
    def named(cls, name: str | None | fd.DynamicName = None, **kwargs) -> Self:
        # we have this option so that we can just plug names in.
        if name is None:
            return cls(**kwargs)
        _size = fd.DynamicName.from_str(
            name, settings=fd.DynamicNameSettings(absolute=True)
            ).capture(nm.FreeNumeric())
        return fd.DynamicName.from_str(name).capture(
            cls(_size=_size, **kwargs)
        )
    
@dataclass(frozen=True)
class RawAxis(Axis): ...

@dataclass(frozen=True)
class StrideMorphism[A:Axis](pc.Morphism[A]):
    _dom: fd.Prod[A]
    _cod_stride: fd.Prod[tuple[A, fd.Prod[nm.Numeric]]]
    name: fd.DynamicName | None = None

    def dom(self) -> pc.ProdObject[A]:
        return pc.ProdObject(self._dom)
    def cod(self) -> pc.ProdObject[A]:
        return pc.ProdObject.from_iter(
            axis for axis, _ in self._cod_stride)
    
    @classmethod
    def from_matrix(cls, 
                    *matrix: fd.Prod[int],
                    dom_names: None | fd.Prod[str] = None,
                    cod_names: None | fd.Prod[str] = None,
                    name: fd.DynamicName | str | None = None):
        matrix = tuple(tuple(row) for row in matrix)
        _cod_stride = tuple(
            (RawAxis.named(cod_names[i] if cod_names is not None else None), 
             tuple(nm.Integer(value) for value in row))
            for i, row in enumerate(matrix)
        )
        _dom = tuple(
            RawAxis.named(dom_names[i] if dom_names is not None else None) 
            for i in range(len(matrix[0])))
        _name = fd.DynamicName.from_str(name)
        return StrideMorphism(_dom=_dom, _cod_stride=_cod_stride, name=_name)