from qiskit.circuit import Instruction

from gates.r_haner.modular_exponentiation import modular_exponentiation_gate, controlled_modular_multiplication_gate
from implementations.regev import Regev


class HanerRegev(Regev):
    def _get_aux_register_size(self, n: int) -> int:
        return n + 1

    @property
    def _prefix(self) -> str:
        return 'Haner'

    def _modular_exponentiation_gate(self, constant: int, N: int, n: int, qd: int) -> Instruction:
        return modular_exponentiation_gate(constant, N, n, qd)

    def _modular_multiplication_gate(self, constant: int, N: int, n: int) -> Instruction:
        return controlled_modular_multiplication_gate(constant, N, n)






