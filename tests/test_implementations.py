import unittest

from ddt import ddt, idata, unpack
from qiskit import Aer, ClassicalRegister
from qiskit.utils import QuantumInstance

from implementations.beauregard import BeauregardShor
from implementations.haner import HanerShor
from implementations.mix import MixShor
from implementations.takahashi import TakahashiShor

implementations_list = [MixShor, BeauregardShor, TakahashiShor, HanerShor]
circuit_types = [False, True]


@ddt
class TestShor(unittest.TestCase):

    def setUp(self) -> None:
        backend = Aer.get_backend('qasm_simulator')
        self.instance = QuantumInstance(backend, shots=64)

    @idata([
        [shor_class, 15, 4, semi, [3, 5]]
        for shor_class in implementations_list
        for semi in circuit_types
    ])
    @unpack
    def test_shor_factoring(self, shor_class, n_v, a_v, semi, factors):
        print(shor_class, n_v, a_v, semi, factors)
        self._test_shor_factoring(shor_class, n_v, a_v, semi, factors)

    @idata([
        [shor_class, 21, 8, semi, [3, 7]]
        for shor_class in implementations_list
        for semi in circuit_types
    ])
    @unpack
    def test_shor_factoring_bigger_numbers(self, shor_class, n_v, a_v, semi, factors):
        print(shor_class, n_v, a_v, semi, factors)
        self._test_shor_factoring(shor_class, n_v, a_v, semi, factors)

    def _test_shor_factoring(self, shor_class, n_v, a_v, semi, factors):
        shor = shor_class(self.instance)
        result = shor.factor(a_v, n_v, semi)
        self.assertCountEqual(list(result), factors)

    @idata([
        [shor_class, n_v, a_v, semi, order]
        for n_v, a_v, order in [(15, 4, 2), (15, 7, 4)]
        for shor_class in implementations_list
        for semi in circuit_types
    ])
    @unpack
    def test_getting_order(self, shor_class, n_v, a_v, semi, order):
        print(shor_class, n_v, a_v, semi, order)
        self._test_getting_order(shor_class, n_v, a_v, semi, order)

    @idata([
        [shor_class, n_v, a_v, semi, order]
        for n_v, a_v, order in [(17, 8, 8), (21, 13, 2)]
        for shor_class in implementations_list
        for semi in circuit_types
    ])
    @unpack
    def test_getting_order_with_bigger_numbers(self, shor_class, n_v, a_v, semi, order):
        print(shor_class, n_v, a_v, semi, order)
        self._test_getting_order(shor_class, n_v, a_v, semi, order)

    def _test_getting_order(self, shor_class, n_v, a_v, semi, order):
        shor = shor_class(self.instance)
        result = shor.get_order(N=n_v, a=a_v, semi_classical=semi)

        if result.order is None:
            self.assertEqual(result.successful_counts, 0)
        else:
            self.assertEqual(result.order, order)
            self.assertGreaterEqual(result.total_counts, result.successful_counts)

    @idata([
        [shor_class, n_v, a_v, semi, order]
        for n_v, a_v, order in [(15, 4, 2), (15, 7, 4)]
        for shor_class in implementations_list
        for semi in circuit_types
    ])
    @unpack
    def test_quantum_circuit(self, shor_class, n_v, a_v, semi, order):
        print(shor_class, n_v, a_v, semi, order)
        self._test_quantum_circuit(shor_class, n_v, a_v, semi, order)

    @idata([
        [shor_class, n_v, a_v, semi, order]
        for n_v, a_v, order in [(17, 8, 8), (21, 13, 2)]
        for shor_class in implementations_list
        for semi in circuit_types
    ])
    @unpack
    def test_quantum_circuit_with_bigger_numbers(self, shor_class, n_v, a_v, semi, order):
        print(shor_class, n_v, a_v, semi, order)
        self._test_quantum_circuit(shor_class, n_v, a_v, semi, order)

    def _test_quantum_circuit(self, shor_class, n_v, a_v, semi, order):
        shor = shor_class(self.instance)
        circuit = shor.construct_circuit(N=n_v, a=a_v, semi_classical=semi, measurement=True)

        result = shor.quantum_instance.execute(circuit)
        counts = result.get_counts(circuit)

        # calculate values that could be measured
        values = [i << (2 * n_v.bit_length() - order.bit_length() + 1) for i in range(order)]

        for measurement in list(counts.keys()):
            print(measurement)
            measurement = shor._parse_measurement(measurement, semi)
            self.assertTrue(measurement in values)

    @idata([
        [shor_class, n_v, a_v, semi, order]
        for n_v, a_v, order in [(15, 4, [1, 4]), (15, 7, [1, 4, 7, 13])]
        for shor_class in implementations_list
        for semi in circuit_types
    ])
    @unpack
    def test_exponentiation_result(self, shor_class, n_v, a_v, semi, values):
        print(shor_class, n_v, a_v, semi, values)
        self._test_exponentiation_result(shor_class, n_v, a_v, semi, values)

    @idata([
        [shor_class, n_v, a_v, semi, order]
        for n_v, a_v, order in [(21, 5, [1, 4, 5, 16, 17, 20]), (25, 4, [1, 4, 6, 9, 11, 14, 16, 19, 21, 24])]
        for shor_class in implementations_list
        for semi in circuit_types
    ])
    @unpack
    def test_exponentiation_result_with_bigger_numbers(self, shor_class, n_v, a_v, semi, values):
        print(shor_class, n_v, a_v, semi, values)
        self._test_exponentiation_result(shor_class, n_v, a_v, semi, values)

    def _test_exponentiation_result(self, shor_class, n_v, a_v, semi, values):
        shor = shor_class(self.instance)

        include_measurement = semi
        circuit = shor.construct_circuit(a_v, n_v, semi_classical=semi, measurement=include_measurement)
        # modify circuit to measure output (down) register
        down_qreg = circuit.qregs[1]
        down_creg = ClassicalRegister(len(down_qreg), name='m')
        circuit.add_register(down_creg)
        circuit.measure(down_qreg, down_creg)

        result = shor.quantum_instance.execute(circuit)
        counts = result.get_counts(circuit)

        measurements = [
            int(key.split(' ')[0], base=2) if semi else int(key, base=2)
            for key in counts.keys()
        ]

        for measurement in measurements:
            self.assertTrue(measurement in values)
