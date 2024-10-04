from typing import Dict, Callable
from unittest import TestCase

from tests.haner.gate_experiments.incrementer import IncrementerExperiment, Experiment
from tests.gate_test import GateTest


class IncrementerTest(GateTest):

    def __init__(self, experiment: Callable[[int], IncrementerExperiment], n: int, test_case: TestCase):
        super().__init__(experiment(n), test_case)
        self._n = n

    def _set_up(self, initial_values: Dict[str, int]) -> None:
        pass

    @property
    def _computations(self) -> Dict[str, Callable[[int], int]]:
        return {
            'x': lambda x: (x + 1) % (1 << self._n)
        }


class Test(IncrementerTest):

    def __init__(self, n: int, test_case: TestCase):
        super().__init__(Experiment, n, test_case)
