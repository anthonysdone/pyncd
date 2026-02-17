from __future__ import annotations
from typing import Callable, Iterable, Iterator, overload
import data_structure.Term as fd
import data_structure.Category as cat
import construction_helpers.product as chp
import construction_helpers.lift as chl
import utilities.utilities as util
from enum import Enum

type AxialObject = cat.ProdObject[cat.RawAxis] | cat.ProdObject[cat.Array[cat.Datatype, cat.RawAxis]]

def get_axes(
    objs: AxialObject
):
    return (
        axis 
        for obj in objs 
        for axis in 
        (obj.shape() 
         if isinstance(obj, cat.Array) 
         else (obj,))
    )

def align_axes(
        left: AxialObject,
        right: AxialObject,
        ctx: fd.Context,
):
    left_axes = tuple(get_axes(left))
    right_axes = tuple(get_axes(right))
    if len(left_axes) != len(right_axes):
        raise ValueError("Cannot align axes of different lengths")
    for left_axis, right_axis in zip(left_axes, right_axes):
        ctx.append_iter((left_axis, right_axis))
    return ctx

def align_composed(
    *targets: cat.BroadcastedCategory | cat.StrideCategory,
    ctx: fd.Context | None = None
):
    targets_expanded = tuple(
        member for m in targets for member in (
            m.content if isinstance(m, cat.Composed) else (m,)
        )
    )
    ctx = ctx or fd.Context()
    for m in zip(targets_expanded[:-1], targets_expanded[1:]):
        ctx = align_axes(
            m[0].cod(),
            m[1].dom(),
            ctx
        )
    return ctx.apply(cat.Composed(
        targets_expanded
    ))

class ExcessProductSide(Enum):
    TOP = 'TOP'
    BOTTOM = 'BOTTOM'

def slice_side(
    target: tuple | cat.ProdObject,
    amount: int,
    side: ExcessProductSide,
):
    if side == ExcessProductSide.TOP:
        return target[:amount]
    return target[-amount:]

def excess_product[L](
    left: cat.ProdObject[L],
    right: cat.ProdObject[L],
    side: ExcessProductSide = ExcessProductSide.TOP
) -> tuple[cat.ProdObject[L] | None, cat.ProdObject[L] | None]:
    excess_left = len(left) - len(right)
    if excess_left > 0:
        return (None, cat.ProdObject.from_iter(slice_side(left, excess_left, side)))
    elif excess_left < 0:
        return (cat.ProdObject.from_iter(slice_side(right, -excess_left, side)), None)
    return (None, None)

def add_excess_lift[B: cat.Datatype](
    left: cat.BroadcastedCategory[B, cat.RawAxis],
    right: cat.BroadcastedCategory[B, cat.RawAxis],
):
    added_lift = excess_product(
        left.cod()[0].shape(), right.dom()[0].shape())
    if added_lift[0] is not None:
        left = chl.morphism_object_lift(left, added_lift[0])
    elif added_lift[1] is not None:
        right = chl.morphism_object_lift(right, added_lift[1])
    return (left, right)

@overload
def composition[B:cat.Datatype](
    left: cat.BroadcastedCategory[B, cat.RawAxis], 
    right: cat.BroadcastedCategory[B, cat.RawAxis]
    ) -> cat.BroadcastedCategory[B, cat.RawAxis]: ...
@overload
def composition(
    left: cat.StrideCategory[cat.RawAxis],
    right: cat.StrideCategory[cat.RawAxis]
) -> cat.StrideCategory[cat.RawAxis]: ...

def composition( # type: ignore
        left,
        right): 
    if isinstance(left, tuple):
        dom_length = max(left) + 1
        dom = tuple(
            util.iallequals(
                right.dom()[cod_idx]
                for cod_idx in left
                if cod_idx == idx
            )
            for idx in range(dom_length)
        )
        left = cat.Rearrangement(
            mapping=left,
            _dom=dom
        )
    if isinstance(right, tuple):
        right = cat.Rearrangement(
            mapping=right,
            _dom = tuple(left.cod())
        )
    if isinstance(left.cod()[0], cat.Array):
        left, right = add_excess_lift(left, right)
    excess_left, excess_right = excess_product(left.cod(), right.dom(), ExcessProductSide.BOTTOM)
    if excess_left is not None:
        left = chp.morphism_product((left, excess_left))
    elif excess_right is not None:
        right = chp.morphism_product((right, excess_right))
    return align_composed(left, right)

cat.Morphism.__matmul__ = composition  # type: ignore
cat.Morphism.__rmatmul__ = lambda x, y: composition(y, x)  # type: ignore