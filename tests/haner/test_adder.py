import unittest

from tests.haner.gate_tests.adder import Test


class AdderTestCase(unittest.TestCase):
    def test_for_1_qubit(self):
        self._test_for_n_qubits(1, [0, 1])

    def test_for_2_qubits(self):
        self._test_for_n_qubits(2, [0, 1, 2, 3])

    def test_for_3_qubits(self):
        self._test_for_n_qubits(3, [0, 1, 3, 5, 6, 7])

    def test_for_4_qubits(self):
        self._test_for_n_qubits(4, [0, 1, 5, 10, 12, 14, 15])

    def test_for_5_qubits(self):
        self._test_for_n_qubits(5, [0, 1, 8, 17, 25, 30, 31])

    def _test_for_n_qubits(self, n, values):
        params = [{'x': x, 'y': y}
                  for x in values
                  for y in values]
        test = Test(n, self)
        test.run_subtests(params)
