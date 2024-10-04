import unittest
from typing import Dict, Callable

from tests.haner.gate_experiments.modular_exponentiation_gate import Experiment
from tests.gate_test import GateTest


class Test(GateTest):
    def __init__(self, constant: int, N: int, n: int, test_case: unittest.TestCase):
        super().__init__(Experiment(constant, N, n), test_case)
        self._constant = constant
        self._N = N
        self._x: int = 0

    def _set_up(self, initial_values: Dict[str, int]) -> None:
        self._x = initial_values['x']

    @property
    def _computations(self) -> Dict[str, Callable[[int], int]]:
        return {
            'y': lambda _: pow(self._constant, self._x, mod=self._N)
        }
