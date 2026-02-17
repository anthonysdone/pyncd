from __future__ import annotations
from typing import Callable, Iterable, Iterator, overload
import data_structure.Term as fd
import data_structure.Category as cat
import construction_helpers.product as chp
import data_structure.Operators as ops

'''
We define:
 Axes >> (Datatype | Array)           -> Array
 StrideCategory >> (Datatype | Array) -> Broadcasted
 StrideCategory >> Broadcasted        -> Broadcasted
 Axes >> BroadcastedCategory          -> BroadcastedCategory
'''

def object_object_lift[B:cat.Datatype, A:cat.Axis](
    base: chp.ProductObjectTarget[cat.Array[B, A], B],
    lift_by: chp.ProductObjectTarget[A]
):
    base_object = chp.object_product(base, conversion=chp.datatype_converter)
    lift_by_object = chp.object_product(lift_by)
    return cat.ProdObject.from_iter(
        cat.Array(
            datatype=segment.datatype, 
            _shape=(*lift_by_object, *segment._shape)
        ) for segment in base_object
    )

def object_morphism_lift[B:cat.Datatype, A:cat.Axis](
    base: chp.ProductObjectTarget[cat.Array[B, A], B],
    lift_by: chp.ProductMorphismTarget[A, cat.StrideCategory[A]]
):
    base_object = chp.object_product(base, conversion=chp.datatype_converter)
    lift_by_morphism = chp.morphism_product(lift_by)
    return chp.morphism_product(
        tuple(ops.Identity.template(segment, lift_by_morphism)
        for segment in base_object)
    )

def morphism_object_lift[B:cat.Datatype, A:cat.Axis](
    base: chp.ProductMorphismTarget[cat.Array[B, A], cat.Broadcasted[B, A], B],
    lift_by: chp.ProductObjectTarget[A]
) -> cat.BroadcastedCategory[B, A]:
    base_morphism = chp.morphism_product(base, conversion=chp.datatype_converter)
    lift_by_object = chp.object_product(lift_by)
    match base_morphism:
        case cat.Block(body=body):
            return base_morphism.reconstruct(
                body=morphism_object_lift(body, lift_by_object)
            )
        case cat.Rearrangement(mapping=mapping, _dom=_dom):
            return cat.Rearrangement(
                mapping=mapping,
                _dom=tuple(object_object_lift(base_morphism.dom(), lift_by_object))
            )
        case cat.Composed(content=ms) | cat.ProductOfMorphisms(content=ms):
            return type(base_morphism).from_iter(
                morphism_object_lift(segment, lift_by_object)
                for segment in ms
            )
        case cat.Broadcasted():
            return broadcasted_stride_lift(base_morphism, lift_by_object)
        
def broadcasted_stride_lift[B:cat.Datatype, A:cat.Axis](
    base: cat.Broadcasted[B, A],
    lift_by: chp.ProductMorphismTarget[A, cat.StrideCategory[A]]
):
    lift_by_morphism = chp.morphism_product(lift_by)
    input_weaves = tuple(
        cat.Weave(
            weave.datatype,
            (cat.WeaveMode.TILED,) * len(lift_by_morphism.cod()) + weave._shape
        )
        for weave in base.input_weaves
    )
    output_weaves = tuple(
        cat.Weave(
            weave.datatype,
            (cat.WeaveMode.TILED,) * len(lift_by_morphism.cod()) + weave._shape
        )
        for weave in base.output_weaves
    )
    reindexings = tuple(
        chp.morphism_product((lift_by_morphism, reindexing))
        for reindexing in base.reindexings
    )
    return base.reconstruct(
        input_weaves=input_weaves,
        output_weaves=output_weaves,
        reindexings=reindexings
    )

@overload
def dynamic_object_lift[B:cat.Datatype, A:cat.Axis](
    base: chp.ProductObjectTarget[cat.Array[B, A], B],
    lift_by: chp.ProductObjectTarget[A]
) -> cat.ProdObject[cat.Array[B, A]]: ...
@overload
def dynamic_object_lift[B:cat.Datatype, A:cat.Axis](
    base: chp.ProductObjectTarget[cat.Array[B, A], B],
    lift_by: cat.StrideCategory[A]
) -> cat.BroadcastedCategory[B, A]: ...

def dynamic_object_lift[B:cat.Datatype, A:cat.Axis]( # type: ignore
    base: chp.ProductObjectTarget[cat.Array[B, A], B],
    lift_by: chp.ProductObjectTarget[A] | cat.StrideCategory[A]
):
    base = chp.general_product(base, conversion=chp.datatype_converter) # type: ignore
    lift_by = chp.general_product(lift_by)
    match base, lift_by:
        case cat.ProdObject(), cat.ProdObject():
            return object_object_lift(base, lift_by)
        case cat.ProdObject(), cat.Morphism():
            return object_morphism_lift(base, lift_by)
        case cat.Broadcasted(), _:
            return broadcasted_stride_lift(base, lift_by)
        case cat.Morphism(), cat.ProdObject():
            return morphism_object_lift(base, lift_by)
        case _:
            raise TypeError("Invalid types for dynamic_object_lift")

cat.Morphism.__rrshift__    = morphism_object_lift # type: ignore
cat.Broadcasted.__rrshift__ = broadcasted_stride_lift # type: ignore
cat.Datatype.__rrshift__    = dynamic_object_lift # type: ignore
cat.Array.__rrshift__       = dynamic_object_lift # type: ignore
cat.ProdObject.__rrshift__  = dynamic_object_lift # type: ignore
