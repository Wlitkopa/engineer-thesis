import unittest

from tests.haner.gate_experiments.cccx import Experiment
from tests.gate_test import GateTest
from utils.typing_ import ValuesMap, ComputationsMap


class Test(GateTest):

    def __init__(self, test_case: unittest.TestCase):

        super().__init__(Experiment(), test_case)
        self._ctrl: int = 0b00

    def _set_up(self, initial_values: ValuesMap) -> None:
        self._ctrl = initial_values['ctrl']

    @property
    def _computations(self) -> ComputationsMap:
        return {
            'x': lambda x: (x + 1) % 2 if self._ctrl == 0b111 else x
        }
