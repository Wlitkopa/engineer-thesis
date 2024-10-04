import unittest

from tests.haner.gate_tests.double_controlled_constant_modulo_adder import Test


class CCControlledConstantModuloAdderTestCase(unittest.TestCase):
    def test_for_1_qubit(self):
        n, constant, x, N = 1, 0, 0, 1
        with self.subTest(constant=constant, N=N, x=x):
            params = {'ctrl': 3, 'x': x, 'g': 1}

            test = Test(constant, N, n, self)
            test.run(params)

    def test_control_register_for_1_qubit(self):
        self._test_control_register_for_n_qubits(1, 0, 0, 1, 1)

    def test_dirty_register_for_1_qubit(self):
        self._test_dirty_register_for_n_qubits(1, 0, 0, 1, [0, 1])

    def test_for_2_qubits(self):
        self._test_for_n_qubits(2, [0, 1, 2, 3], [2, 3])

    def test_control_register_for_2_qubits(self):
        self._test_control_register_for_n_qubits(2, 2, 1, 3, 1)

    def test_dirty_register_for_2_qubits(self):
        self._test_dirty_register_for_n_qubits(2, 2, 1, 3, [0, 1])

    def test_for_3_qubits(self):
        self._test_for_n_qubits(3, [0, 1, 3, 5, 6, 7], [5, 6, 7])

    def test_control_register_for_3_qubits(self):
        self._test_control_register_for_n_qubits(3, 5, 4, 6, 2)

    def test_dirty_register_for_3_qubits(self):
        self._test_dirty_register_for_n_qubits(3, 5, 4, 6, [0, 1, 2, 3])

    def test_for_4_qubits(self):
        self._test_for_n_qubits(4, [0, 1, 5, 10, 12, 14, 15], [11, 15])

    def test_control_register_for_4_qubits(self):
        self._test_control_register_for_n_qubits(4, 12, 7, 15, 5)

    def test_dirty_register_for_4_qubits(self):
        self._test_dirty_register_for_n_qubits(4, 12, 7, 15, [0, 1, 5, 6, 7])

    def test_for_5_qubits(self):
        self._test_for_n_qubits(5, [0, 1, 8, 17, 25, 30, 31], [21, 23, 25])

    def test_control_register_for_5_qubits(self):
        self._test_control_register_for_n_qubits(5, 14, 8, 21, 10)

    def test_dirty_register_for_5_qubits(self):
        self._test_dirty_register_for_n_qubits(5, 14, 8, 21, [0, 1, 5, 10, 14, 15])

    def _test_for_n_qubits(self, n, values, N_values):
        for N in N_values:
            current_values = [value for value in values if value < N]
            for constant in current_values:
                with self.subTest(constant=constant, N=N):
                    params = [{'ctrl': 3, 'x': x, 'g': 1} for x in current_values]

                    test = Test(constant, N, n, self)
                    test.run_subtests(params)

    def _test_dirty_register_for_n_qubits(self, n, constant, x, N, g_values):
        params = [{'ctrl': 3, 'x': x, 'g': g} for g in g_values]
        test = Test(constant, N, n, self)
        test.run_subtests(params)

    def _test_control_register_for_n_qubits(self, n, constant, x, N, g):
        ctrl_values = [0, 1, 2, 3]
        params = [{'ctrl': ctrl, 'x': x, 'g': g} for ctrl in ctrl_values]
        test = Test(constant, N, n, self)
        test.run_subtests(params)
