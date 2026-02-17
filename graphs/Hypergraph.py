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
    Iterable,
)
import random
import math
from abc import ABC, abstractmethod
from enum import Enum
from itertools import product
from collections import Counter
from construction_helpers import product as chp

import data_structure.Term as fd # for 'foundations'
import data_structure.Numeric as nm
import utilities.utilities as util
import data_structure.Category as cat

# We track the location of a graph morphism
# within the constructor morphism.
type Location = fd.Prod[
    tuple[Type[cat.Composed] | Type[cat.ProductOfMorphisms], int]
    | cat.BlockTag
]

@dataclass(frozen=True)
class GraphObj[L](fd.UTerm):
    @classmethod
    def generate(cls, target: cat.ProdObject[L]):
        return tuple(GraphObj() for _ in target) # type: ignore

@dataclass(frozen=True)
class GraphRoot[L, M: cat.Morphism](fd.Term):
    dom: fd.Prod[GraphObj[L]]
    cod: fd.Prod[GraphObj[L]]
    location: Location
    wraps: M
    def blocks(self) -> fd.Prod[cat.BlockTag]:
        return tuple(
            tag for tag in self.location
            if isinstance(tag, cat.BlockTag)
        )
    def primary_block(self) -> cat.BlockTag | None:
        blocks = self.blocks()
        return blocks[0] if blocks else None
    def roots(self) -> fd.Prod[GraphRoot[L, M]]:
        return (self,)



def morphism_to_hypergraph[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M],
    location: Location = (),
    domain: None | fd.Prod[GraphObj[L]] = None,
) -> Hypergraph[L, M]:
    domain = domain or GraphObj.generate(target.dom())
    match target:
        case cat.Block(body=body, block_tag=block_tag):
            subgraph = morphism_to_hypergraph(
                body, (*location, block_tag), domain
            )
            roots = subgraph.roots
            codomain = subgraph.cod
        case cat.Composed(content=ms):
            roots = ()
            codomain = domain
            for i, m in enumerate(ms):
                subgraph = morphism_to_hypergraph(
                    m, (*location, (cat.Composed, i)), codomain
                )
                roots = (*roots, *subgraph.roots)
                codomain = subgraph.cod
        case cat.Rearrangement():
            roots = ()
            codomain = target.apply(domain)
        case cat.ProductOfMorphisms(content=ms):
            roots = ()
            codomain = ()
            for i, (m, dom) in enumerate(target.partition(domain)):
                subgraph = morphism_to_hypergraph(
                    m, (*location, (cat.ProductOfMorphisms, i)), dom
                )
                roots = (*roots, *subgraph.roots)
                codomain = (*codomain, *subgraph.cod)
        case _:
            codomain = GraphObj.generate(target.cod())
            root = GraphRoot(
                dom=domain,
                cod=codomain,
                wraps=target,
                location=location
            )
            roots = (root,)
    return Hypergraph(
        dom=domain,
        cod=codomain,
        wraps=target,
        roots=roots
    )

def replace_morphism[L, M: cat.Morphism](
    target: cat.ProdCategory[L, M],
    location: Location,
    replacement: cat.ProdCategory[L, M],
) -> cat.ProdCategory[L, M]:
    match target, location:
        case _, ():
            return replacement
        case cat.Block(body=body, block_tag=block_tag), (cat.BlockTag() as tag, *rest):
            assert tag == block_tag
            return cat.Block(
                body=replace_morphism(body, tuple(rest), replacement),
                block_tag=block_tag
            )
        case (cat.Composed(content=ms), ((cat.Composed, i), *rest)) | \
            (cat.ProductOfMorphisms(content=ms), ((cat.ProductOfMorphisms, i), *rest)):
            content = (
                *ms[:i], replace_morphism(ms[i], tuple(rest), replacement), *ms[i+1:]
            )
            return type(target)(content=content)
    raise KeyError(f'Location {location} not found in target morphism {target}')
        
@dataclass(frozen=True)
class Hypergraph[L, M: cat.Morphism](fd.Term):
    dom: fd.Prod[GraphObj[L]]
    cod: fd.Prod[GraphObj[L]]
    wraps: cat.ProdCategory[L, M]
    roots: fd.Prod[GraphRoot[L, M]]

    @classmethod
    def from_morphism(cls, target: cat.ProdCategory[L, M]):
        return morphism_to_hypergraph(target)
    
################################
## PROCESSING OF A HYPERGRAPH ##
################################

class DomCod(Enum):
    DOM = 'DOM'
    COD = 'COD'

@dataclass(frozen=True)
class GraphEdge[L, M: cat.Morphism]:
    dom_cod: DomCod
    segment: int
    graph_morphism: GraphRoot[L, M]

    def node(self) -> GraphObj[L]:
        match self.dom_cod:
            case DomCod.DOM:
                return self.graph_morphism.dom[self.segment]
            case DomCod.COD:
                return self.graph_morphism.cod[self.segment]
            
    def obj(self) -> L:
        match self.dom_cod:
            case DomCod.DOM:
                return self.graph_morphism.wraps.dom()[self.segment]
            case DomCod.COD:
                return self.graph_morphism.wraps.cod()[self.segment]

def get_edges[L, M: cat.Morphism](
    hypergraph: Hypergraph[L, M],
):
    edges: dict[GraphObj[L], list[GraphEdge[L, M]]] = {}
    for graph_morphism in hypergraph.roots:
        edge_info = (
            *product((DomCod.DOM,), enumerate(graph_morphism.dom)),
            *product((DomCod.COD,), enumerate(graph_morphism.cod))
        )
        for dom_cod, (i, obj_node) in edge_info:
            edge = GraphEdge(
                dom_cod=dom_cod,
                segment=i,
                graph_morphism=graph_morphism
            )
            edges[obj_node] = edges.get(obj_node, []) + [edge]
    return {k: tuple(v) for k, v in edges.items()}

@dataclass
class ProcessedHypergraph[L, M: cat.Morphism]:
    hypergraph: Hypergraph[L, M]
    edges: dict[GraphObj[L], fd.Prod[GraphEdge[L, M]]]

    @classmethod
    def from_morphism(cls, target: cat.ProdCategory[L, M]):
        hypergraph = Hypergraph.from_morphism(target)
        edges = get_edges(hypergraph)
        return cls(
            hypergraph=hypergraph,
            edges=edges
        )
    def obj(self, node: GraphObj[L]) -> L:
        return util.iallequals(
            edge.obj() for edge in self.edges[node]
        )
    @classmethod
    def from_hypergraph(cls, hypergraph: Hypergraph[L, M]):
        edges = get_edges(hypergraph)
        return cls(
            hypergraph=hypergraph,
            edges=edges
        )