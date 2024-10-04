from typing import Tuple, List, Callable, Dict

Size = int
Name = str

QRegsSpec = Dict[Name, Size]

Value = int
Computation = Callable[[Value], Value]
ComputationsMap = Dict[Name, Computation]
ValuesMap = Dict[Name, Value]

del Tuple, List, Callable, Dict
