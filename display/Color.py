from __future__ import annotations
from typing import Any, Callable, Iterable, Self
from abc import ABC
from dataclasses import dataclass, field

def original(s: str) -> str:
    start = s.find('\x1b[')
    if start == -1:
        return s
    end = s.find('m', start) + 1
    return s[:start] + original(s[end:])

def colored_output(target: str, fg: Color | None = None, bg: Color | None = None):
    # From https://stackoverflow.com/questions/74589665/how-to-print-rgb-colour-to-the-terminal
    # print(fg, bg)
    def to_256(x: float) -> int:
        return max(0, min(255, int(x * 256)))
    fg_rgb = fg.rgb256() if fg else None
    bg_rgb = bg.rgb256() if bg else None
    match fg_rgb, bg_rgb:
        case None, None:
            return target
        case None, _:
            return f"\033[48;2;{';'.join(map(str, bg_rgb))}m{target}\033[0m"
        case _, None:
            return f"\033[38;2;{';'.join(map(str, fg_rgb))}m{target}\033[0m"
        case _, _:
            return f"\033[38;2;{';'.join(map(str, fg_rgb))}m\033[48;2;{';'.join(map(str, bg_rgb))}m{target}\033[0m"
        
@dataclass(frozen=True)
class Color(ABC):
    def red(self) -> float: ...
    def green(self) -> float: ...
    def blue(self) -> float: ...
    def chroma(self) -> float:
        rgb = (self.red(), self.green(), self.blue())
        return max(rgb) - min(rgb)
    def rgb256(self) -> tuple[int, int, int]:
        def to_256(x: float) -> int:
            return max(0, min(255, int(x * 256)))
        return (
            to_256(self.red()),
            to_256(self.green()),
            to_256(self.blue())
        )
    @classmethod
    def white(cls) -> Self: ...
    @classmethod
    def black(cls) -> Self: ...
    @classmethod
    def from_rgb(cls, r01: float, g01: float, b01: float) -> Self: ...
    @classmethod
    def from_hue(cls, hue360: float) -> Self:
        sextant = (hue360 % 360) / 60
        def clip(x: float) -> float:
            return max(0, min(1, x))
        return cls.from_rgb(
            clip(abs(sextant-3)-1),
            clip(2-abs(sextant-4)),
            clip(2-abs(sextant-2))
        )

    def contrast(self) -> Self:
        if self.luminance() < 0.5:
            return self.white()
        else:
            return self.black()
    def luminance(self) -> float:
        return 0.2126 * self.red() + 0.7152 * self.green() + 0.0722 * self.blue()
    def __call__(self, target: str, bg: Color | None = None) -> str:
        return colored_output(target, fg=self, bg=bg)
    def bg(self, target: str, fg: Color | None = None) -> str:
        fg = fg or self.contrast()
        col_out = colored_output(target, fg=fg, bg=self)
        return col_out

@dataclass(frozen=True)
class HexadecimalColor(Color):
    value: str
    def red(self) -> float:
        return float(int(self.value[1:3], 16)) / 255
    def green(self) -> float:
        return float(int(self.value[3:5], 16)) / 255
    def blue(self) -> float:
        return float(int(self.value[5:7], 16)) / 255
    
    @classmethod
    def from_rgb(cls, r01: float, g01: float, b01: float) -> Self:
        r = max(0, min(255, int(r01 * 256)))
        g = max(0, min(255, int(g01 * 256)))
        b = max(0, min(255, int(b01 * 256)))
        return cls(f'#{r:02x}{g:02x}{b:02x}')
        
    @classmethod
    def white(cls) -> Self:
        return cls('#ffffff')
    @classmethod
    def black(cls) -> Self:
        return cls('#000000')
    @classmethod
    def from_int(cls, value: int) -> Self:
        return cls(f'#{value:06x}'[:7])
    

