from typing import Dict, Callable
from unittest import TestCase

from tests.haner.gate_experiments.adder import AdderExperiment, Experiment
from tests.gate_test import GateTest


class AdderTest(GateTest):

    def __init__(self, experiment: Callable[[int], AdderExperiment], n: int, test_case: TestCase):
        super().__init__(experiment(n), test_case)
        self._n = n
        self._x: int

    def _set_up(self, initial_values: Dict[str, int]) -> None:
        self._x = initial_values['x']

    @property
    def _computations(self) -> Dict[str, Callable[[int], int]]:
        return {
            'y': lambda y: (y + self._x) % (1 << self._n)
        }


class Test(AdderTest):

    def __init__(self, n: int, test_case: TestCase):
        super().__init__(Experiment, n, test_case)
