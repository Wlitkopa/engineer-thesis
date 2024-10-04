from typing import Union

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Instruction, Gate, ParameterVector
from qiskit.circuit.library import QFT

from gates.beauregard.constant_adder import phi_constant_adder, get_angles


def modular_exponentiation_gate(constant: int, N: int, n: int) -> Instruction:
    up_qreg = QuantumRegister(2 * n, name='up')
    down_qreg = QuantumRegister(n, name='down')
    aux_qreg = QuantumRegister(n + 2, name='aux')

    circuit = QuantumCircuit(up_qreg,
                             down_qreg,
                             aux_qreg,
                             name=f'{constant}^x mod {N}')

    qft = QFT(n + 1, do_swaps=False).to_gate()
    iqft = qft.inverse()

    phi_add_N = phi_constant_adder(get_angles(N, n + 1))
    iphi_add_N = phi_add_N.inverse()
    c_phi_add_N = phi_add_N.control(1)

    for i in range(2 * n):
        partial_constant = pow(constant, pow(2, i), mod=N)
        modulo_multiplier = controlled_modular_multiplication_gate(partial_constant, N, n, c_phi_add_N, iphi_add_N, qft, iqft)
        circuit.append(modulo_multiplier, [up_qreg[i], *down_qreg, *aux_qreg])

    return circuit.to_instruction()


def controlled_modular_multiplication_gate(a: int, N: int, n: int,
                                           c_phi_add_N: Gate, iphi_add_N: Gate, qft: Gate, iqft: Gate) -> Instruction:
    ctrl_qreg = QuantumRegister(1, 'ctrl')
    x_qreg = QuantumRegister(n, 'x')
    b_qreg = QuantumRegister(n + 1, 'b')
    flag_qreg = QuantumRegister(1, 'flag')

    circuit = QuantumCircuit(ctrl_qreg,
                             x_qreg,
                             b_qreg,
                             flag_qreg,
                             name='cmult_a_mod_N')

    angle_params = ParameterVector('angles', length=n + 1)
    modulo_adder = _double_controlled_phi_add_mod_N(angle_params, c_phi_add_N, iphi_add_N, qft, iqft)

    def append_adder(adder: QuantumCircuit, constant: int, idx: int):
        partial_constant = (pow(2, idx, mod=N) * constant) % N
        angles = get_angles(partial_constant, n + 1)
        bound = adder.assign_parameters({angle_params: angles})
        circuit.append(bound, [*ctrl_qreg, x_qreg[idx], *b_qreg, *flag_qreg])

    circuit.append(qft, b_qreg)

    for i in range(n):
        append_adder(modulo_adder, a, i)

    circuit.append(iqft, b_qreg)

    for i in range(n):
        circuit.cswap(ctrl_qreg, x_qreg[i], b_qreg[i])

    circuit.append(qft, b_qreg)

    a_inv = pow(a, -1, mod=N)
    modulo_adder_inv = modulo_adder.inverse()
    for i in reversed(range(n)):
        append_adder(modulo_adder_inv, a_inv, i)

    circuit.append(iqft, b_qreg)

    return circuit.to_instruction()


def _double_controlled_phi_add_mod_N(angles: Union[np.ndarray, ParameterVector],
                                     c_phi_add_N: Gate,
                                     iphi_add_N: Gate,
                                     qft: Gate,
                                     iqft: Gate
                                     ) -> QuantumCircuit:
    ctrl_qreg = QuantumRegister(2, 'ctrl')
    b_qreg = QuantumRegister(len(angles), 'b')
    flag_qreg = QuantumRegister(1, 'flag')

    circuit = QuantumCircuit(ctrl_qreg,
                             b_qreg,
                             flag_qreg,
                             name='ccphi_add_a_mod_N')

    cc_phi_add_a = phi_constant_adder(angles).control(2)
    cc_iphi_add_a = cc_phi_add_a.inverse()

    circuit.append(cc_phi_add_a, [*ctrl_qreg, *b_qreg])

    circuit.append(iphi_add_N, b_qreg)

    circuit.append(iqft, b_qreg)
    circuit.cx(b_qreg[-1], flag_qreg[0])
    circuit.append(qft, b_qreg)

    circuit.append(c_phi_add_N, [*flag_qreg, *b_qreg])

    circuit.append(cc_iphi_add_a, [*ctrl_qreg, *b_qreg])

    circuit.append(iqft, b_qreg)
    circuit.x(b_qreg[-1])
    circuit.cx(b_qreg[-1], flag_qreg[0])
    circuit.x(b_qreg[-1])
    circuit.append(qft, b_qreg)

    circuit.append(cc_phi_add_a, [*ctrl_qreg, *b_qreg])

    return circuit
