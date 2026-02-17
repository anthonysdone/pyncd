import data_structure.Term as fd
import data_structure.Category as cat
import data_structure.Numeric as nm
from dataclasses import dataclass, field

@dataclass
class ConfigLog:
    terms: set[fd.Term] = field(default_factory=set)

    def add_term(self, term: fd.Term):
        self.terms.add(term)

    def search_for_type[T: fd.Term](self, target_type: type[T], term: fd.GeneralTerm):
        if isinstance(term, target_type):
            self.add_term(term)
            return
        fd.deep_iterate(term, lambda t: self.search_for_type(target_type, t)) # type: ignore

@dataclass
class NumericConfig:
    config_log: ConfigLog = field(default_factory=ConfigLog)
    imposed_context: fd.Context = field(default_factory=fd.Context)

    def incorporate(self, target: cat.Morphism):
        self.config_log.search_for_type(nm.FreeNumeric, target)

    def search(self, string: str) -> set[nm.FreeNumeric]:
        return {
            term for term in self.config_log.terms
            if isinstance(term, nm.FreeNumeric)
            and isinstance(term.uid._name, fd.DynamicName)
            and term.uid._name.body == string
        }
    
    def assign(self, targets: str | nm.FreeNumeric | set[nm.FreeNumeric], value: int):
        if isinstance(targets, str):
            targets = self.search(targets)
        elif isinstance(targets, nm.FreeNumeric):
            targets = {targets}
        bucket = fd.EqualityClass[nm.Numeric](
            nm.Numeric, 
            set(term.uid for term in targets), 
            nm.Integer(value)
        )
        self.imposed_context.equality_classes.append(bucket)

    def apply[T: fd.GeneralTerm](self, target: T) -> T:
        return self.imposed_context.apply(target)
        

    


    