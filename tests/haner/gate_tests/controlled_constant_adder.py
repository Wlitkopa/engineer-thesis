import unittest
from typing import Dict, Callable

from tests.gate_test import GateTest

from tests.haner.gate_experiments.controlled_constant_adder import Experiment


class Test(GateTest):
    def __init__(self, constant: int, n: int, test_case: unittest.TestCase):
        super().__init__(Experiment(constant, n), test_case)
        self._constant = constant
        self._n = n
        self._ctrl: int

    def _set_up(self, initial_values: Dict[str, int]) -> None:
        self._ctrl = initial_values['ctrl']

    @property
    def _computations(self) -> Dict[str, Callable[[int], int]]:
        return {
            'x': lambda x: (x + self._constant) % (1 << self._n) if self._ctrl == 0b1 else x
        }
