import unittest

from tests.haner.gate_tests.incrementer import Test


class IncrementerTestCase(unittest.TestCase):
    def test_for_1_qubit(self):
        self._test_for_n_qubits(1, [0, 1], 1)

    def test_dirty_register_for_1_qubits(self):
        self._test_dirty_register_for_n_qubits(1, 1, [0, 1])

    def test_for_2_qubits(self):
        self._test_for_n_qubits(2, [0, 1, 2, 3], 1)

    def test_dirty_register_for_2_qubits(self):
        self._test_dirty_register_for_n_qubits(2, 2, [0, 1, 2, 3])

    def test_for_3_qubits(self):
        self._test_for_n_qubits(3, [0, 1, 3, 5, 6, 7], 2)

    def test_dirty_register_for_3_qubits(self):
        self._test_dirty_register_for_n_qubits(3, 5, [0, 1, 5, 6, 7])

    def test_for_4_qubits(self):
        self._test_for_n_qubits(4, [0, 1, 5, 10, 12, 14, 15], 5)

    def test_dirty_register_for_4_qubits(self):
        self._test_dirty_register_for_n_qubits(4, 12, [0, 1, 5, 10, 14, 15])

    def test_for_5_qubits(self):
        self._test_for_n_qubits(5, [0, 1, 8, 17, 25, 30, 31], 10)

    def test_dirty_register_for_5_qubits(self):
        self._test_dirty_register_for_n_qubits(5, 25, [0, 1, 11, 21, 30, 31])

    def _test_for_n_qubits(self, n, x_values, g):
        params = [{'x': x, 'g': g} for x in x_values]

        test = Test(n, self)
        test.run_subtests(params)

    def _test_dirty_register_for_n_qubits(self, n, x, g_values):
        params = [{'x': x, 'g': g} for g in g_values]
        test = Test(n, self)
        test.run_subtests(params)
