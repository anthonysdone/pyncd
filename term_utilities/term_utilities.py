from __future__ import annotations
from enum import Enum
from typing import Any
import utilities.utilities as util
import data_structure.Category as cat
import data_structure.Term as fd

class WhichCategory(Enum):
    UNCLEAR = 'UNCLEAR'
    BROADCASTED = 'BROADCASTED'
    STRIDE = 'STRIDE'

type CategoricalConstruct = (
    cat.StrideCategory | cat.BroadcastedCategory | cat.ProdObject
    | cat.Array | cat.Datatype | cat.Axis
)

def identify_category(
        target: CategoricalConstruct) -> WhichCategory:
    if isinstance(target, (cat.Datatype, cat.Array, cat.Broadcasted)):
        return WhichCategory.BROADCASTED
    if isinstance(target, cat.StrideMorphism):
        return WhichCategory.STRIDE
    match target:
        case cat.Block(body=body):
            return identify_category(body)
        case ((cat.ProdObject() as parts)
              | cat.ProductOfMorphisms(content=parts) 
              | cat.Composed(content=parts) 
              | cat.Rearrangement(_dom=parts)):
            possibilites = (
                category
                for segment in parts
                if (category := identify_category(segment)))
            return util.iallequals(possibilites, WhichCategory.UNCLEAR)
        case _:
            raise ValueError(f"Cannot identify category of {target}")
        
type Mappable[L=Any] = (cat.Rearrangement[L] 
                 | cat.ProductOfMorphisms[L, Mappable[L]] 
                 | cat.Composed[L, Mappable[L]]
                 | cat.Block[L, Mappable[L]])

def is_mappable_broadcast(
    target: cat.Broadcasted
):
    return all(is_mappable(eta) for eta in target.reindexings)

def is_mappable(
    target: cat.ProdCategory[Any, Any]
):
    match target:
        case cat.Rearrangement():
            return True
        case cat.Composed(content=ms) | cat.ProductOfMorphisms(content=ms):
            return all(is_mappable(segment) for segment in ms)
        case cat.Block(body=body):
            return is_mappable(body)
        case _:
            return False
        
def get_mapping(
    target: cat.ProdCategory[Any, Any]
) -> fd.Prod[int]:
    if not is_mappable(target):
        raise ValueError(f"Target {target} is not mappable")
    match target:
        case cat.Rearrangement():
            return target.mapping
        case cat.ProductOfMorphisms():
            offset = 0
            mapping = ()
            for segment in target.content:
                new_mapping = get_mapping(segment)
                mapping = (*mapping, *(i + offset for i in new_mapping))
                offset += len(segment.dom())
            return mapping
        case cat.Composed():
            mapping = get_mapping(target.content[0])
            for segment in target.content[1:]:
                mapping = tuple(mapping[i] for i in get_mapping(segment))
            return mapping
        case cat.Block(body=body):
            return get_mapping(body)
        case _:
            raise ValueError(f"Target {target} is not mappable")

def is_identity(
    target: cat.ProdCategory[Any, Any]
) -> bool:
    match target:
        case cat.Rearrangement():
            return target.mapping == tuple(range(len(target.dom())))
        case cat.ProductOfMorphisms(content=ms) | cat.Composed(content=ms):
            return all(is_identity(segment) for segment in ms)
        case cat.Block(body=body):
            return is_identity(body)
        case _:
            return False