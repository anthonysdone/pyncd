from __future__ import annotations
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Type

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import utilities.utilities as util
import data_structure.Category as cat
import construction_helpers.product as chp

@dataclass(frozen=True)
class HypergraphObject[L](fd.UTerm):
    obj: L = None # type: ignore
    @classmethod
    def template(cls, target: cat.ProdObject[L]) -> fd.Prod[HypergraphObject[L]]:
        return tuple(HypergraphObject(obj=obj) for obj in target) # type: ignore
    
@dataclass(frozen=True)
class Hypergraph[L, M: cat.Morphism](fd.UTerm, ABC):
    dom: fd.Prod[HypergraphObject[L]] = ()
    cod: fd.Prod[HypergraphObject[L]] = ()
    blocks: fd.Prod[cat.BlockTag] = ()

    @abstractmethod
    def subgraphs(self) -> fd.Prod[Hypergraph[L, M]]: ...

@dataclass(frozen=True)
class Multigraph[L, M: cat.Morphism](Hypergraph[L, M]):
    _subgraphs: fd.Prod[Hypergraph[L, M]] = ()

    def subgraphs(self):
        return self._subgraphs

@dataclass(frozen=True)
class HypergraphRoot[L, M: cat.Morphism](Hypergraph[L, M]):
    wraps: M = None # type: ignore

    @classmethod
    def template(cls, 
                 target: M, 
                 dom: fd.Prod[HypergraphObject[L]] | None = None, 
                 blocks: fd.Prod[cat.BlockTag] = ()) -> HypergraphRoot[L, M]:
        return cls(
            uid=fd.UID(HypergraphRoot),
            dom=dom or HypergraphObject.template(target.dom()),
            cod=HypergraphObject.template(target.cod()),
            blocks=blocks,
            wraps=target
        )
    
    def subgraphs(self):
        return (self,)

type LocationUnit = tuple[Type[cat.Composed] | Type[cat.ProductOfMorphisms], int] | cat.BlockTag
type Location = fd.Prod[LocationUnit]

@dataclass(frozen=True)
class StructuredGraph[L, M: cat.Morphism](Hypergraph[L, M]):
    ''' A structured graph maintains a reference to the original expression,
    enabling replacement to interact with the structure of the initial term. '''
    wraps: cat.ProdCategory[L, M] = None # type: ignore
    _subgraphs: fd.Prod[tuple[Location, HypergraphRoot[L, M] | StructuredGraph[L, M]]] = ()

    def subgraphs(self) -> fd.Prod[Hypergraph[L, M]]:
        return tuple(subgraph for _, subgraph in self._subgraphs)

    def append_location(self, location: Location):
        return self.reconstruct(_subgraphs=tuple(
            ((*location, *loc), root)
            for loc, root in self._subgraphs
        ))

    @classmethod
    def from_morphism(
        cls,
        target: cat.ProdCategory[L, M],
        dom: fd.Prod[HypergraphObject[L]] | None = None,
        location: Location = ()) -> StructuredGraph[L, M]:
        dom = dom or HypergraphObject.template(target.dom())
        subgraphs: fd.Prod[tuple[Location, HypergraphRoot[L, M] | StructuredGraph[L, M]]] = ()
        cod: fd.Prod[HypergraphObject[L]] = ()
        match target:
            case cat.Block():
                location = (*location, target.block_tag)
                subgraph = cls.from_morphism(target.body, dom, location)
                subgraphs = ((location, subgraph),)
                cod = subgraph.cod
            case cat.Composed(ms):
                subgraphs = ()
                cod = dom
                for i, m in enumerate(ms):
                    subgraph = cls.from_morphism(m, cod, 
                                                 (*location, (cat.Composed, i)))
                    cod = subgraph.cod
                    subgraphs = (*subgraphs, *subgraph._subgraphs)
            case cat.ProductOfMorphisms(ms):
                subgraphs = ()
                for i, (m, _dom) in enumerate(target.partition(dom)):
                    subgraph = cls.from_morphism(m, _dom, 
                                                 (*location, (cat.ProductOfMorphisms, i)))
                    subgraphs = (*subgraphs, *subgraph._subgraphs)
                    cod = (*cod, *subgraph.cod)
            case cat.Rearrangement():
                subgraphs = ()
                cod = target.apply(dom)
            case _:
                subgraph = HypergraphRoot.template(
                    target, dom=dom, 
                    blocks=tuple(l for l in location if isinstance(l, cat.BlockTag))
                )
                subgraphs = ((location, subgraph),)
                cod = subgraph.cod
        return cls(
            uid=fd.UID(StructuredGraph),
            dom=dom,
            cod=cod,
            blocks=tuple(l for l in location if isinstance(l, cat.BlockTag)),
            wraps=target,
            _subgraphs=subgraphs
        )

def morphism2hypergraph[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M]) -> StructuredGraph[L, M]:
    return StructuredGraph.from_morphism(target)

############################
## HYPERGRAPH -> MORPHISM ##
############################
## This assumes that blocks are in their own Hypergraphs ##

def remove_prefix[S](prefix: fd.Prod[S], target: fd.Prod[S]) -> fd.Prod[S]:
    assert target[:len(prefix)] == prefix
    return target[len(prefix):]

def get_body[L, M: cat.Morphism](block: cat.BlockTag, target: Hypergraph[L, M]):
    match target:
        case HypergraphRoot():
            return target.reconstruct(blocks=remove_prefix((block,), target.blocks))
        case _:
            return Multigraph(
                uid=fd.UID(Multigraph),
                dom=target.dom,
                cod=target.cod,
                blocks=remove_prefix((block,), target.blocks),
                _subgraphs=tuple(
                    get_body(block, subgraph) for subgraph in target.subgraphs()
                )
            )

def hypergraph2morphism[L, M: cat.Morphism](
    graph: Hypergraph[L, M]
) -> cat.ProdCategory[L, M]:
    if graph.blocks:
        block = graph.blocks[0]
        body = get_body(block, graph)
        return cat.Block(
            block_tag=block,
            body=hypergraph2morphism(body)
        )
    if isinstance(graph, HypergraphRoot):
        return graph.wraps
    sequence: fd.Prod[cat.ProdCategory[L, M]] = ()
    remaining = graph
    while remaining.subgraphs():
        sequence_front, remaining = cleave(remaining)
        sequence = (*sequence, *sequence_front)
        if not remaining.subgraphs() and remaining.dom == remaining.cod:
            break
    if len(sequence) == 1:
        return sequence[0]
    return cat.Composed(sequence)


def cleave[L, M: cat.Morphism](
    target: Hypergraph[L, M],
) -> tuple[fd.Prod[cat.ProdCategory[L, M]], Hypergraph[L, M]]:
    dom = list(target.dom)
    assert(len(dom) == len(set(dom)))
    subgraphs = target.subgraphs()
    cleaved_graphs, remaining_graphs = util.predicate_partition(
        subgraphs,
        lambda graph: set(graph.dom) <= set(dom)
    )
    future_dom = set(target.cod).union(*(graph.dom for graph in remaining_graphs))
    kept_obj = util.intersection(dom, future_dom)
    column_dom = util.concat(cleave.dom for cleave in cleaved_graphs) + kept_obj
    column_cod = util.concat(cleave.cod for cleave in cleaved_graphs) + kept_obj

    mapping = [dom.index(obj) for obj in column_dom]
    
    rearrangement = cat.Rearrangement(
        mapping = tuple(mapping),
        _dom = tuple(hypergraph_obj.obj for hypergraph_obj in dom)
    ) if mapping != list(range(len(dom))) else None

    column: cat.ProdCategory[L, M] = chp.morphism_product(
        (*(hypergraph2morphism(graph) for graph in cleaved_graphs),
         *(obj.obj for obj in kept_obj))
    ) # type: ignore

    multigraph = Multigraph(
        uid=fd.UID(Multigraph),
        dom=column_cod,
        cod=target.cod,
        blocks=target.blocks,
        _subgraphs=tuple(remaining_graphs)
    )

    return (
        (rearrangement, column) if rearrangement else (column,),
        multigraph
    )