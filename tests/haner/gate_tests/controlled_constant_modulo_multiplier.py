import unittest
from typing import Dict, Callable

from tests.haner.gate_experiments.controlled_constant_modulo_multiplier import Experiment
from tests.gate_test import GateTest


class Test(GateTest):
    def __init__(self, constant: int, N: int, n: int, test_case: unittest.TestCase):
        super().__init__(Experiment(constant, N, n), test_case)
        self._constant = constant
        self._N = N
        self._x: int
        self._ctrl: int

    def _set_up(self, initial_values: Dict[str, int]) -> None:
        self._x = initial_values['x']
        self._ctrl = initial_values['ctrl']

    @property
    def _computations(self) -> Dict[str, Callable[[int], int]]:
        return {
            'y': lambda _: (self._x * self._constant) % self._N if self._ctrl == 0b1 else 0,
        }
