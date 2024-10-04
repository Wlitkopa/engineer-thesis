import unittest

from tests.haner.gate_tests.modular_exponentiation_gate import Test


class ModuloExponentiationGateTestCase(unittest.TestCase):

    def test_for_3_qubits(self):
        self._test_for_n_qubits(3, 2, 3, [0, 1, 29, 42, 62, 63])

    def test_for_4_qubits(self):
        self._test_for_n_qubits(4, 3, 5, [0, 1, 42, 113, 254, 255])

    def test_for_5_qubits(self):
        self._test_for_n_qubits(5, 3, 7, [0, 1, 42, 301, 1022, 1023])

    def _test_for_n_qubits(self, n, first, second, x_values):
        N = first * second
        constant_values = [constant for constant in range(2, N)
                           if constant % first != 0 and constant % second != 0]

        for constant in constant_values:
            with self.subTest(constant=constant, N=N):
                test = Test(constant, N, n, self)
                params = [{'x': x, 'y': 1} for x in x_values]
                test.run_subtests(params)
