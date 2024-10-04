import unittest
from abc import ABC, abstractmethod
from typing import List, Dict, Callable

from qiskit.result import Result

from tests.gate_experiment import GateExperiment

NAME = 'test_experiment'


class GateTest(ABC):
    def __init__(self, experiment: GateExperiment, test_case: unittest.TestCase):
        self._experiment = experiment
        self.test_case = test_case

    def run_subtests(self, params: List[Dict[str, int]]):
        for initial_values in params:
            with self.test_case.subTest(initial_values):
                self.run(initial_values)

    def run(self, initial_values: Dict[str, int]):
        self._set_up(initial_values)
        result = self._experiment.run(initial_values, NAME)

        computations = self._computations
        for name, value in zip(self._experiment.qreg_names, self._parse_result(result, NAME)):
            initial_value = initial_values.get(name, 0)

            if name in computations:
                expected_value = computations[name](initial_value)
            else:
                expected_value = initial_value

            self.test_case.assertEqual(value, expected_value,
                                       f'Wrong value in register "{name}" (expected {expected_value}, got {value})')

    def _parse_result(self, result: Result, name: str) -> List[int]:
        output = list(result.get_counts(name).keys())
        self.test_case.assertEqual(len(output), 1)

        values_as_str = output[0].split(' ')
        values = [int(value, 2) for value in values_as_str]
        return values[::-1]

    @abstractmethod
    def _set_up(self, initial_values: Dict[str, int]) -> None:
        pass

    @property
    @abstractmethod
    def _computations(self) -> Dict[str, Callable[[int], int]]:
        pass
