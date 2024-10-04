from typing import Dict, Callable
from unittest import TestCase

from tests.haner.gate_experiments.controlled_incrementer import Experiment
from tests.haner.gate_tests.incrementer import IncrementerTest


class Test(IncrementerTest):

    def __init__(self, n: int, test_case: TestCase):
        super().__init__(Experiment, n, test_case)
        self._ctrl: int

    def _set_up(self, initial_values: Dict[str, int]) -> None:
        super()._set_up(initial_values)
        self._ctrl = initial_values['ctrl']

    @property
    def _computations(self) -> Dict[str, Callable[[int], int]]:
        computations = super()._computations
        return {
            'x': lambda x: computations['x'](x) if self._ctrl == 0b1 else x
        }
