from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, overload
import utilities.utilities as util
import data_structure.Term as fd
import data_structure.Category as cat
import construction_helpers.product as chp
import construction_helpers.lift as chl
from enum import Enum
import display.Box as Box
import display.Color as Color
import term_utilities.term_utilities as tutils
from abc import ABC, abstractmethod

@dataclass
class DisplayCategory[L, M:cat.Morphism](ABC):
    opposite: bool
    segment_separator: Box.Box | None
    
    @abstractmethod
    def display_lone(self, target: L) -> Box.Box: ...

    def display_morphism(self, target: M) -> Box.Box:
        return self.display_morphism_default(target)

    def display_block(self, target: cat.Block[L, cat.ProdCategory[L, M]]) -> Box.Box:
        body = self.display_category(target.body)
        title_box = ()
        description_box = ()
        if target.aesthetics is not None:
            title_box: fd.Prod[Box.Box] = (Box.TextBox(target.aesthetics.title),) \
                if target.aesthetics.title is not None else ()
            description_box: fd.Prod[Box.Box] = (Box.TextBox(target.aesthetics.description),) \
                if target.aesthetics.description is not None else ()
            return Box.Padded(Box.Vertical(
                title_box + description_box + (body,)),
                pad_horizontal=1,
                pad_vertical=1,
            )
        body = Box.Vertical(
            title_box + description_box + (body,)
        ) if title_box or description_box else body
        return Box.Padded(
            body,
            pad_horizontal=1,
            pad_vertical=1,
        )

    def display_rearrangement(self, target: cat.Rearrangement[L]) -> Box.Box:
        content = (
            self.display_prod_object(target.dom()),
            Box.TextBox('=>=' if not self.opposite else '=<='),
            self.display_prod_object(target.cod())
        )
        return Box.Horizontal.from_iter(content)

    def display_prod_object(self, target: cat.ProdObject[L]) -> Box.Box:
        return Box.Vertical.from_iter(util.join_with_none(
            (self.display_lone(segment) for segment in target),
            self.segment_separator
        ))
    
    def display_morphism_default(self, target: cat.ProdCategory[L, M]) -> Box.Box:
        content = (
            self.display_prod_object(target.dom()),
            Box.TextBox('=>=' if not self.opposite else '=<='),
            self.display_prod_object(target.cod())
        )
        return Box.Horizontal.from_iter(content)
    
    def display_category_unreversed(self, target: cat.ProdCategory[L, M]) -> Box.Box:
        match target:
            case cat.Composed(content=ms):
                return Box.Horizontal.from_iter(
                    self.display_category_unreversed(segment)
                    for segment in ms
                )
            case cat.ProductOfMorphisms(content=ms):
                return Box.Vertical.from_iter(
                    util.join_with_none(
                        (self.display_category_unreversed(segment)
                        for segment in ms),
                        self.segment_separator
                    )
                )
            case cat.Rearrangement():
                return self.display_rearrangement(target)
            case cat.Block():
                return self.display_block(target)
            case _:
                return self.display_morphism(target)
            
    def display_category(self, target: cat.ProdCategory[L, M]) -> Box.Box:
        target_box = self.display_category_unreversed(target)
        if self.opposite:
            return Box.reverse_box(target_box)
        return target_box
    
def display_uterm(target: fd.UTerm) -> Box.Box:
    text = ''
    if target.uid._name is not None:
        text = target.uid._name.to_bodies()[:2]
    else:
        text = f'{target.uid._id:X}'[-2:]
    return Box.TextBox(text)

@dataclass
class DisplayStrideCategory[A:cat.Axis](DisplayCategory[A, cat.StrideMorphism[A]]):
    def display_morphism_short(self, target: cat.StrideMorphism[A], axis_width: int = 4) -> Box.Box:
        content = (
            Box.Vertical.from_iter(
                self.display_axis(axis, axis_widths=(axis_width,0))
                for axis in target.cod()
            ),
            Box.TextBox('<'),
            Box.Vertical.from_iter(
                self.display_axis(axis, axis_widths=(0,axis_width))
                for axis in target.dom()
            ))
        return Box.Horizontal.from_iter(content)
    
    def display_axis(self, target: A, axis_widths: tuple[int, int] = (0,0)) -> Box.Box:
        return Box.Horizontal((
            Box.Fill('─', min_width=axis_widths[0], min_height=1),
            display_uterm(target),
            Box.Fill('─', min_width=axis_widths[1], min_height=1)
        ))
    
    def display_rearrangement(self, target: cat.Rearrangement[A]) -> Box.Box:
        axis_width = 4
        return Box.Horizontal((
            Box.Vertical.from_iter(
                Box.Fill('─', min_width=axis_width, min_height=1)
                for axis in target.cod()
            ),
            Box.Vertical.from_iter(
                Box.Horizontal((
                    Box.TextBox(f'[{index}]'),
                    Box.Fill('─', min_width=axis_width, min_height=1))
                ) for index in target.mapping
            )
        ))
    
    def display_lone(self, target: A) -> Box.Box:
        return display_uterm(target)
    
stride = DisplayStrideCategory(
    opposite=True,
    segment_separator=None
)

def display_datatype(target: cat.Datatype) -> Box.Box:
    return Box.TextBox(type(target).__qualname__[:2])

@dataclass
class DisplayBroadcastedCategory[B:cat.Datatype, A:cat.Axis](
    DisplayCategory[cat.Array[B, A], cat.Broadcasted[B, A]]):
    display_stride: DisplayStrideCategory[A]
    display_datatype: bool

    def display_lone(self, target: cat.Array[B, A]) -> Box.Box:
        return Box.Vertical.from_iter((
            *(self.display_stride.display_lone(axis) for axis in target._shape),
            *((display_datatype(target.datatype),) if self.display_datatype else ())
            ))
    
    def display_weave(self, target: cat.Weave[B, A], reindexing: cat.StrideCategory[A]):
        width = 4
        reindexing_cod = iter(reindexing.cod())
        return Box.Vertical(
            (*(
                Box.Horizontal(
                    (self.display_stride.display_lone(axis), 
                    Box.Fill(' ', min_width=width, min_height=1))
                )
                if isinstance(axis, cat.Axis) else
                Box.Horizontal(
                    (self.display_stride.display_lone(next(reindexing_cod)),
                    Box.Fill(' ', min_width=width, min_height=1))
                )
                for axis in target._shape
            ), 
            *((display_datatype(target.datatype),) 
              if self.display_datatype else ()))
            )
    
    def display_morphism(self, target: cat.Broadcasted[B, A]) -> Box.Box:
        box_text = f' >{type(target.operator).__qualname__}> '

        # content = (
        #     self.display_prod_object(target.dom()),
        #     Box.Vertical(
        #         (
        #             Box.Horizontal((
        #                 Box.Fill('─', min_width=len(box_text)//2),
        #                 self.display_stride.display_prod_object(target.degree()),
        #                 Box.Fill('─', min_width=len(box_text)//2)
        #             )),
        #             Box.TextBox(box_text)
        #         )
        #     ),
        #     self.display_prod_object(target.cod())
        # )
        content = (
            self.display_prod_object(target.dom()),
            Box.Vertical((
                *(
                    self.display_stride.display_morphism(reindexing)
                    for reindexing in target.reindexings
                ),
                Box.TextBox(box_text)
            )),
            self.display_prod_object(target.cod())
        )
        return Box.Horizontal.from_iter(content)
            
broadcasted = DisplayBroadcastedCategory(
    opposite=False,
    segment_separator=Box.Fill('-', min_height=1),
    display_stride=stride,
    display_datatype=True
)

def display(target: tutils.CategoricalConstruct) -> str:
    which_category = tutils.identify_category(target)
    display_engine = broadcasted \
        if which_category == tutils.WhichCategory.BROADCASTED else stride
    match target:
        case cat.Morphism():
            return str(display_engine.display_category(target)) # type: ignore
        case cat.ProdObject():
            return str(display_engine.display_prod_object(target)) # type: ignore
        case _:
            raise ValueError(f"Cannot display target of type {type(target)}")