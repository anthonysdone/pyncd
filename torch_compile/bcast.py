from typing import Sequence, Iterable, Any, Callable, Literal
import data_structure.Category as cat
import term_utilities.term_utilities as tutil
import utilities.utilities as util

############################
## BROADCASTING SEMANTICS ##
############################

def no_swaps(target: Sequence[int]) -> bool:
    return all(target[i+1] >= target[i] for i in range(len(target) - 1))

def unsqueeze_guide[T](
        degree_size: int,
        mapping: tuple[int, ...],
        imprint: Sequence[T]):
    # Think of the degree size as:
    # (0, 1, 2)
    # And the mapping might be:
    # (0,    2)
    # This should instruct us to make;
    # (p, 1, r)
    # By rehaping with a one
    return tuple(
        imprint[i] if i in mapping else 1
        for i in range(degree_size)
    )

def is_semantically_broadcastable(target: cat.Broadcasted) -> bool:
    return (
        all(
            weave.target().shape() == cat.ProdObject()
            for weave in (
                *target.input_weaves,
                *target.output_weaves
            )
        )
        and
        all(
            tutil.is_mappable(eta)
            and no_swaps(tutil.get_mapping(eta))
            for eta in target.reindexings
        )
    )


##########
## DIMS ##
##########

def weave_displacement(target: cat.Weave) -> None | int:
    section: Literal['left'] | Literal['center'] | Literal['right'] = 'left'
    dim = -1
    for axis_weave in target._shape:
        match section, axis_weave:
            case 'left', cat.Axis:
                section = 'center'
            case 'center', cat.WeaveMode:
                section = 'right'
            case 'right', cat.WeaveMode:
                dim -= 1
            case 'right', cat.Axis:
                return None
    return dim

def get_displacement[T](target: cat.Broadcasted) -> None | int:
    if not all(tutil.is_identity(eta) for eta in target.reindexings):
        return None
    weaves = target.input_weaves + target.output_weaves
    displacement = util.iallequals(
        (weave_displacement(weave) for weave in weaves),
        None
    )
    return displacement
        
##########
## VMAP ##
##########

def no_copying[T](target: Sequence[T]):
    return len(target) == len(set(target))

def vmappable(target: cat.Broadcasted):
    return all(
        tutil.is_mappable(eta)
        and no_copying(tutil.get_mapping(eta))
        for eta in target.reindexings
    )

def find_vmap_loc(current_degree: int, target: Iterable[cat.Axis | int]):
    location = 0
    target = iter(target)
    for axis_mapping in target:
        if axis_mapping == current_degree:
            return location
        if isinstance(axis_mapping, cat.Axis) or current_degree < axis_mapping:
            location += 1

def broadcast_vmap(target: cat.Broadcasted[Any, cat.Axis]):
    assert vmappable(target)
    degree_idxs = list(range(len(target.degree())))
    reindexing_mappings = tuple(
        tutil.get_mapping(reindexing)
        for reindexing in target.reindexings
    )
    # We can't do reindex copying with vmap.
    assert all(len(mapping) == len(set(mapping)) for mapping in reindexing_mappings)
    # We insert integers where the degree joins the weave.
    input_weaves_mapped = [
        input_weave.imprint(mapping)
        for input_weave, mapping
        in zip(target.input_weaves, reindexing_mappings)
    ]
    output_weaves_mapped = [
        output_weave.imprint(degree_idxs)
        for output_weave
        in target.output_weaves
    ]
    # Build vmap from the bottom up
    for current_degree in reversed(degree_idxs):
        input_loc = tuple(
            find_vmap_loc(current_degree, input_weave)
            for input_weave in input_weaves_mapped
        )
        output_loc: tuple[int, ...] = tuple(
            find_vmap_loc(current_degree, output_weave)
            for output_weave in output_weaves_mapped
        ) # type: ignore
        yield input_loc, output_loc