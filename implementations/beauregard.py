from qiskit import QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.circuit.library import QFT

from gates.beauregard.modular_exponentiation import modular_exponentiation_gate, controlled_modular_multiplication_gate
from gates.beauregard.constant_adder import phi_constant_adder, get_angles
from implementations.shor import Shor


class BeauregardShor(Shor):
    def _construct_circuit_with_semiclassical_QFT(self, a: int, N: int, n: int) -> QuantumCircuit:
        self._qft = QFT(n + 1, do_swaps=False).to_gate()
        self._iqft = self._qft.inverse()

        phi_add_N = phi_constant_adder(get_angles(N, n + 1))
        self._iphi_add_N = phi_add_N.inverse()
        self._c_phi_add_N = phi_add_N.control(1)

        return super()._construct_circuit_with_semiclassical_QFT(a, N, n)

    def _get_aux_register_size(self, n: int) -> int:
        return n + 2

    @property
    def _prefix(self) -> str:
        return 'Beauregard'

    def _modular_exponentiation_gate(self, constant: int, N: int, n: int) -> Instruction:
        return modular_exponentiation_gate(constant, N, n)

    def _modular_multiplication_gate(self, constant: int, N: int, n: int) -> Instruction:
        return controlled_modular_multiplication_gate(constant, N, n, self._c_phi_add_N, self._iphi_add_N, self._qft,
                                                      self._iqft)
