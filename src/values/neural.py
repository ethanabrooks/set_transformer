from dataclasses import dataclass

from train.value_unconditional import compute_values
from values.tabular import Values as TabularValues


@dataclass(frozen=True)
class Values(TabularValues):
    @classmethod
    def compute_values(cls, *args, **kwargs):
        return compute_values(*args, **kwargs)
