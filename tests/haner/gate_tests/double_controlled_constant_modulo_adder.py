import unittest
from typing import Dict, Callable

from tests.gate_test import GateTest
from tests.haner.gate_experiments.double_controlled_constant_modulo_adder import Experiment


class Test(GateTest):
    def __init__(self, constant: int, N: int, n: int, test_case: unittest.TestCase):
        super().__init__(Experiment(constant, N, n), test_case)
        self._constant = constant
        self._N = N
        self._ctrl: int = 0b00

    def _set_up(self, initial_values: Dict[str, int]) -> None:
        self._ctrl: int = initial_values['ctrl']

    @property
    def _computations(self) -> Dict[str, Callable[[int], int]]:
        return {
            'x': lambda x: (x + self._constant) % self._N if self._ctrl == 0b11 else x,
        }
