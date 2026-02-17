from __future__ import annotations
from typing import Callable, Iterable, Iterator, overload
import data_structure.Term as fd
import data_structure.Category as cat
from enum import Enum

def partition_string(
    target: str,
    separator: str = ','
):
    return [
        segment.strip()
        for segment in target.strip().split(separator)
    ]

def signature_to_buckets(
    target: str,
    space_seperated: bool = True,
):
    input_portion, output_portion = target.split('->')
    output_processing = [
        (name, -i-1) 
        for i, name in 
        enumerate(partition_string(output_portion, ' '))]
    # Reindexed
    names_indexes = dict(output_processing)
    input_indexes: list[list[int]] = []
    index_pointer = 0
    for i, segment in enumerate(partition_string(input_portion, ',')):
        input_indexes.append([])
        for ik, name in enumerate(partition_string(segment, ' ')):
            if name not in names_indexes:
                names_indexes[name] = index_pointer
                index_pointer += 1
            key = names_indexes[name]
            input_indexes[i].append(key)

    return (
        input_indexes, 
        [index for _, index in output_processing],
        dict((v, k) for k, v in names_indexes.items())
    )

def bucketed_to_broadcast[B:cat.Datatype=cat.Reals](
    input_indexes: list[list[int]],
    output_indexes: list[int],
    indexes_names: dict[int, str] = {},
    datatype: B = cat.Reals()
):
    axes = {
        index: (
            cat.RawAxis.named(name) 
            if (name := indexes_names.get(index)) is not None
            else cat.RawAxis()
        )
        for index in set.union(
            set(output_indexes),
            *(set(segment) for segment in input_indexes)
        )
    }
    input_weaves = tuple(
        cat.Weave(
            datatype, tuple(
                axes[index] if 0 <= index else cat.WeaveMode.TILED
                for index in segment)
        )
        for segment in input_indexes
    )
    output_weaves = (cat.Weave(
        datatype, tuple(cat.WeaveMode.TILED for _ in output_indexes)
    ),)
    degree = tuple(axes[index] for index in output_indexes)
    reindexings = tuple(
        cat.Rearrangement(tuple(
            -index-1 for index 
            in segment if index < 0
        ), _dom=degree)
        for segment in input_indexes
    )
    signature = tuple(tuple(
        index for index in segment
        if index >= 0
    ) for segment in input_indexes)
    return signature, input_weaves, output_weaves, reindexings