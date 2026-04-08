from typing import (
    Callable,
    Iterator,
    Type,
    Iterable
)
from enum import Enum

type Prod[T] = tuple[T, ...]
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

def deconcatenate(target: Prod[int], dom_length: int | None = None) -> Prod[tuple[int, int]]:
    dom_length = dom_length or max(target) + 1
    return tuple(
        (L0, R0)
        for L0 in range(len(target))
        for R0 in range(dom_length)
        if all((k < L0) == (target[k] < R0) for k in range(len(target)))
    )[1:]

def concat[T, Y=T](xss: Iterable[Iterable[T]], func: Callable[[T], Y] = lambda x: x):
    return tuple(
        func(x)
        for xs in xss
        for x in xs
    )

def intersection[T](xs: Iterable[T], ys: Iterable[T]) -> Prod[T]:
    set_ys = set(ys)
    return tuple(x for x in xs if x in set_ys)

def predicate_partition[T](xs: Iterable[T], predicate: Callable[[T], bool]) -> tuple[Prod[T], Prod[T]]:
    true_part: list[T] = []
    false_part: list[T] = []
    for x in xs:
        (true_part if predicate(x) else false_part).append(x)
    return tuple(true_part), tuple(false_part)
