import unittest

from tests.haner.gate_tests.controlled_adder import Test


class CAdderTestCase(unittest.TestCase):
    def test_for_1_qubit(self):
        self._test_for_n_qubits(1, [0, 1])

    def test_control_register_for_1_qubit(self):
        self._test_control_register_for_n_qubits(1, 1, 1)

    def test_for_2_qubits(self):
        self._test_for_n_qubits(2, [0, 1, 2, 3])

    def test_control_register_for_2_qubits(self):
        self._test_control_register_for_n_qubits(2, 2, 1)

    def test_for_3_qubits(self):
        self._test_for_n_qubits(3, [0, 1, 3, 5, 6, 7])

    def test_control_register_for_3_qubits(self):
        self._test_control_register_for_n_qubits(3, 5, 4)

    def test_for_4_qubits(self):
        self._test_for_n_qubits(4, [0, 1, 5, 10, 12, 14, 15])

    def test_control_register_for_4_qubits(self):
        self._test_control_register_for_n_qubits(4, 12, 7)

    def test_control_register_for_5_qubits(self):
        self._test_control_register_for_n_qubits(5, 28, 8)

    def test_for_5_qubits(self):
        self._test_for_n_qubits(5, [0, 1, 8, 17, 25, 30, 31])

    def _test_for_n_qubits(self, n, values):
        params = [{'ctrl': 1, 'x': x, 'y': y}
                  for x in values
                  for y in values]

        test = Test(n, self)
        test.run_subtests(params)

    def _test_control_register_for_n_qubits(self, n, x, y):
        ctrl_values = [0, 1]
        params = [{'ctrl': ctrl, 'x': x, 'y': y} for ctrl in ctrl_values]
        test = Test(n, self)
        test.run_subtests(params)
