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



if __name__ == "__main__":
    N = 21
    d_ceil_bool = False
    qd_ceil_bool = True
    shots_num = 128
    file_name = "/home/koan/myHome/AGH/PracaInżynierska/pycharm_github/shor_mmik/output_data/regev/quantum_part/ceil_ceil/N_21"
    regev = HanerRegev(shots_num)
    regev.file_data_analyzer(file_name)



