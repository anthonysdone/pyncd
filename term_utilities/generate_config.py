import data_structure.Term as fd
import data_structure.Category as cat
import data_structure.Numeric as nm
import utilities.utilities as util
from dataclasses import dataclass, field
from typing import Callable, Type, Protocol

'''
Each term is a partially constructed term with free variables as inputs.
For configurations, we care about Numeric inputs.
'''

class UIDEquipped(Protocol):
    @property
    def uid(self) -> fd.UID: ...

def deep_pass[T, Y](target: T, func: Callable[[T], fd.Prod[Y]]) -> fd.Prod[Y]:
    match target:
        case fd.Term():
            vals = target.dict().values()
        case tuple():
            vals = target
        case _:
            return func(target)
    return tuple(item for val in vals for item in func(val))

def deep_iterate[T](target: T, func: Callable[[T], None]) -> None:
    match target:
        case fd.Term():
            vals = target.dict().values()
        case tuple():
            vals = target
        case _:
            return None
    for val in vals:
        func(val)

@dataclass
class ConfigLog[T: UIDEquipped]:
    target_type: Type[T]
    terms: list[T] = field(default_factory=list)
    imposed_context: fd.Context = field(default_factory=fd.Context)

    def add_term(self, term: T):
        if term not in self.terms:
            self.terms.append(term)

    def incorporate(self, target: fd.GeneralTerm):
        if isinstance(target, self.target_type):
            self.add_term(target)
            return
        deep_iterate(target, self.incorporate)

    def search(self, target_string: str) -> list[T]:
        return [
            term for term in self.terms
            if term.uid._name.body == target_string # type: ignore
        ]
    
    def assign(self, targets: str, value: T):
        bucket = fd.EqualityClass[T]( #type: ignore
            self.target_type,
            set(term.uid for term in self.search(targets)),
            value
        )
        self.imposed_context.equality_classes.append(bucket)

    def get_bucket(self, target: T) -> None | tuple[int, T]:
        for i, bucket in enumerate(self.imposed_context.equality_classes):
            if target.uid in bucket.bucket:
                return i, bucket.canonical
        return None

    def apply_context[S: fd.GeneralTerm](self, target: S) -> S:
        return self.imposed_context.apply(target)
    
    def __call__[S: fd.GeneralTerm](self, target: S) -> S:
        return self.apply_context(target)
    
@dataclass
class NumericConfig(ConfigLog[nm.FreeNumeric]):
    target_type: Type[nm.FreeNumeric] = nm.FreeNumeric
    
    @classmethod
    def template(cls, target: fd.GeneralTerm):
        config = cls()
        config.incorporate(target)
        return config
    
    def assign_value(self, targets: str, value: int) -> None:
        self.assign(targets, nm.Integer(value)) # type: ignore

    def assign_values(self, **targets: int):
        for target_string, value in targets.items():
            self.assign_value(target_string, value)