from typing import Dict, Callable
from unittest import TestCase

from tests.haner.gate_experiments.controlled_comparator import Experiment
from tests.haner.gate_tests.comparator import ComparatorTest


class Test(ComparatorTest):

    def __init__(self, constant: int, n: int, test_case: TestCase):
        super().__init__(Experiment, constant, n, test_case)
        self._ctrl: int

    def _set_up(self, initial_values: Dict[str, int]) -> None:
        super()._set_up(initial_values)
        self._ctrl = initial_values['ctrl']

    @property
    def _computations(self) -> Dict[str, Callable[[int], int]]:
        computations = super()._computations
        return {
            'c': lambda c: computations['c'](c) if self._ctrl == 0b1 else c
        }
