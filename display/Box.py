from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Iterable, Iterator, overload
from enum import Enum
import display.Color as Color
import utilities.justification as js

type Prod[T] = tuple[T, ...]

@dataclass
class Box(ABC):
    @abstractmethod
    def height(self) -> int: ...
    @abstractmethod
    def width(self) -> int: ...

    def rows(self, width: int | None = None, height: int | None = None) -> Iterator[str]:
        return js.justify(
            (js.justify_str((row,), width or self.width())
             for row in self.raw_rows()),
             ' ' * (width or self.width()),
             height,
        )
    def raw_rows(self) -> Iterator[str]:
        raise NotImplementedError()
    def render(self, width: int | None = None, height: int | None = None) -> str:
        return '\n'.join(
            self.rows(width=width, height=height)
        )
    def __str__(self) -> str:
        return self.render()

def fit_to_length[T](
        target: Iterable[T], 
        length: int | None, 
        fill: T) -> Iterator[T]:
    if length is None:
        yield from target
        return
    counter = 0
    for item in target:
        if length is not None and counter >= length:
            return
        yield item
        counter += 1
    if length is not None and counter < length:
        for _ in range(length - counter):
            yield fill

def fit_length_str(target: Iterable[str], length: int | None, buffer: str = ' ') -> str:
    return ''.join(fit_to_length(target, length, buffer))

@dataclass
class AssociativeBox(Box, ABC):
    content: Prod[Box]
    justify_mode: js.JustifyMode = js.JustifyMode.LEFT
    @classmethod
    def from_iter(cls, target: Iterable[Box], justify_mode: js.JustifyMode = js.JustifyMode.LEFT) -> AssociativeBox:
        return cls(content=tuple(target), justify_mode=justify_mode)
    def __iter__(self) -> Iterator[Box]:
        return iter(self.content)

@dataclass
class Horizontal(AssociativeBox):
    justify_mode: js.JustifyMode = js.JustifyMode.SPREAD
    def height(self) -> int:
        if len(self.content) == 0:
            return 0
        return max(box.height() for box in self.content)
    def width(self) -> int:
        return sum(box.width() for box in self.content)
    def rows(self, width: int | None = None, height: int | None = None):
        
        box_rows: Iterator[Iterator[str]] = (
            column.rows(height=height or self.height())
            for column in self.content
        )
        return js.justify(
            (
                js.justify_str(
                    row_parts, # type: ignore
                    width or self.width(),
                    mode=self.justify_mode)
                for row_parts in zip(*box_rows)
            ),
            ' ' * (width or self.width()),
            height,
            self.justify_mode
        )
    
@dataclass
class Vertical(AssociativeBox):
    def height(self) -> int:
        return sum(box.height() for box in self.content)
    def width(self) -> int:
        if len(self.content) == 0:
            return 0
        return max(box.width() for box in self.content)
    def rows(self, width: int | None = None, height: int | None = None):
        return js.justify(
            (row
             for box in self.content
             for row in box.rows(width=width or self.width(), height=None)),
             ' ' * (width or self.width()),
            height,
            self.justify_mode
        )

@dataclass
class TextBox(Box):
    text: str
    fg: Color.Color | None = None
    bg: Color.Color | None = None
    def height(self) -> int:
        return self.text.count('\n') + 1
    def width(self) -> int:
        if Color.original(self.text) == '':
            return 0
        return max(len(line) for line in Color.original(self.text).split('\n'))
    def raw_rows(self) -> Iterator[str]:
        for line in self.text.split('\n') or ['']:
            yield Color.colored_output(line, fg=self.fg, bg=self.bg)

@dataclass
class PaddedPadding:
    default: str = '*'
    horizontal: str | None = None
    vertical: str | None = None
    top_left: str | None = None
    top_right: str | None = None
    bottom_left: str | None = None
    bottom_right: str | None = None
    
    def pad_row(self, target: str, width: int) -> str:
        horizontal_pad = (self.vertical or self.default) * width
        return horizontal_pad + target + horizontal_pad

    def top_lines(self, width: int, height: int) -> Iterator[str]:
        for h in range(height):
            yield self.pad_row(self.top_line(width - 2 * h), h)

    def bottom_lines(self, width: int, height: int) -> Iterator[str]:
        for h in reversed(range(height)):
            yield self.pad_row(self.bottom_line(width - 2 * h), h)

    def top_line(self, width: int) -> str:
        assert width >= 2
        return (self.top_left or self.default) + (
            (self.horizontal or self.default) * (width - 2)
        ) + (self.top_right or self.default)
    def bottom_line(self, width: int) -> str:
        assert width >= 2
        return (self.bottom_left or self.default) + (
            (self.horizontal or self.default) * (width - 2)
        ) + (self.bottom_right or self.default)
    
unicode_box_padding = PaddedPadding(
    default = '*',
    horizontal = '═',
    vertical = '║',
    top_left = '╔',
    top_right = '╗',
    bottom_left = '╚',
    bottom_right = '╝'
)
    
@dataclass
class Padded(Box):
    body: Box
    pad_horizontal: int = 1
    pad_vertical: int = 1
    padding: PaddedPadding = field(default_factory=lambda: unicode_box_padding)

    def height(self) -> int:
        return self.body.height() + 2 * self.pad_vertical
    def width(self) -> int:
        return self.body.width() + 2 * self.pad_horizontal
    def rows(self, width: int | None = None, height: int | None = None) -> Iterator[str]:
        width = width or self.width()
        height = height or self.height()
        assert width >= 2 * self.pad_horizontal
        assert height >= 2 * self.pad_vertical
        core_width = width - 2 * self.pad_horizontal
        core_height = height - 2 * self.pad_vertical

        yield from self.padding.top_lines(width, self.pad_vertical)
        yield from (
            self.padding.pad_row(row, self.pad_horizontal)
            for row in self.body.rows(width=core_width, height=core_height)
        )
        yield from self.padding.bottom_lines(width, self.pad_vertical)

@dataclass
class Fill(Box):
    fill: str = ' '
    min_width: int = 0
    min_height: int = 0

    def height(self) -> int:
        return self.min_height
    def width(self) -> int:
        return self.min_width
    def rows(self, width: int | None = None, height: int | None = None) -> Iterator[str]:
        fill_row = fit_length_str(
            self.fill, 
            width or self.min_width, 
            self.fill)
        for _ in range(height or self.min_height):
            yield fill_row

def reverse_box(target: Box) -> Box:
    match target:
        case Horizontal(content=boxes):
            return Horizontal.from_iter(
                reverse_box(box) for box in reversed(boxes)
            )
        case Vertical(content=boxes):
            return Vertical.from_iter(
                reverse_box(box) for box in boxes
            )
        case _:
            return target