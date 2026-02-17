from typing import (
    Callable,
    Iterator,
    Type,
    Iterable
)
from enum import Enum

class AllEqualsFallback(Enum):
    RAISE = 'RAISE'

def iallequals[T](xs: Iterable[T], fallback: AllEqualsFallback | T = AllEqualsFallback.RAISE) -> T:
    iterator = iter(xs)
    _first = next(iterator)
    while True:
        try:
            x = next(iterator)
            if x != _first:
                if fallback != AllEqualsFallback.RAISE:
                    return fallback
                raise ValueError(f"Elements are not all equal: {_first} != {x}")
        except StopIteration:
            break
    return _first

def yielder[K, V](target: Iterable[K], generator: Callable[[K], V]):
    previous: dict[K, V] = {}
    for key in target:
        if key not in previous:
            previous[key] = generator(key)
        yield previous[key]

def join_with_none[T](target: Iterable[T], separator: T | None) -> Iterator[T]:
    iterator = iter(target)
    if separator is None:
        yield from iterator
        return
    try:
        next_one = next(iterator)
    except StopIteration:
        return ()
    while True:
        yield next_one
        try:
            next_one = next(iterator)
        except StopIteration:
            break
        yield separator

def unique_iterable[T](target: Iterable[T]) -> Iterator[T]:
    seen: set[T] = set()
    for item in target:
        if item not in seen:
            seen.add(item)
            yield item