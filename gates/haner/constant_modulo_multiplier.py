from itertools import chain

from qiskit.circuit import Gate

from gates.haner.constant_modulo_adder import double_controlled_constant_modulo_adder
from utils.circuit_creation import create_circuit
from utils.typing_ import QRegsSpec


def controlled_constant_modulo_multiplier(constant: int, N: int, n: int) -> Gate:
    if n == 1:
        raise ValueError("Creating circuit for n = 1 not supported")

    circuit = create_circuit(controlled_constant_modulo_multiplier_regs(n), f'CC-Mult_({constant})_Mod_{N}')
    ctrl_qreg, x_qreg, y_qreg, flag_qreg = circuit.qregs

    for i in reversed(range(n)):
        partial_constant = (pow(2, i) * constant) % N

        g_qreg = x_qreg[:]
        g_qreg.pop(i)

        circuit.append(
            double_controlled_constant_modulo_adder(partial_constant, N, n),
            chain(ctrl_qreg, [x_qreg[i]], y_qreg, g_qreg, flag_qreg)
        )

    return circuit.to_gate()


def controlled_constant_modulo_multiplier_regs(n: int) -> QRegsSpec:
    if n == 1:
        raise ValueError("Case n = 1 not supported")

    return {
        'ctrl': 1,
        'x': n,
        'y': n,
        'flag': 1
    }
