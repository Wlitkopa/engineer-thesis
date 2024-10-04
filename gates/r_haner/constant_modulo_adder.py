from itertools import chain

from qiskit.circuit import Gate

from gates.haner.comparator import double_controlled_comparator, double_controlled_comparator_regs
from gates.haner.constant_adder import controlled_constant_adder
from utils.circuit_creation import create_circuit
from utils.typing_ import QRegsSpec


def double_controlled_constant_modulo_adder(constant: int, N: int, n: int) -> Gate:
    circuit = create_circuit(double_controlled_constant_modulo_adder_regs(n), f'CC-Add_({constant})_Mod_{N}')
    ctrl_qreg, x_qreg, g_qreg, flag_qreg = circuit.qregs

    adder_regs = list(chain(flag_qreg, x_qreg, [g_qreg[0]]))

    circuit.append(
        double_controlled_comparator(N - constant, n),
        circuit.qubits
    )
    circuit.append(
        controlled_constant_adder(constant, n),
        adder_regs
    )
    circuit.ccx(ctrl_qreg[0], ctrl_qreg[1], flag_qreg[0])
    circuit.append(
        _controlled_constant_subtractor(N - constant, n),
        adder_regs
    )
    circuit.append(
        double_controlled_comparator(constant, n),
        circuit.qubits
    )

    return circuit.to_gate()


def _controlled_constant_subtractor(constant: int, n: int) -> Gate:
    adder = controlled_constant_adder(constant, n)
    return adder.inverse()


def double_controlled_constant_modulo_adder_regs(n: int) -> QRegsSpec:
    spec = double_controlled_comparator_regs(n)
    return {(name if name != 'c' else 'flag'): size for (name, size) in spec.items()}
