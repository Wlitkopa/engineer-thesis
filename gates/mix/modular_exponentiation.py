from itertools import chain

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import QFT

from gates.beauregard.constant_adder import get_angles, phi_constant_adder
from gates.mix.comparator import double_controlled_comparator


def modular_exponentiation_gate(constant: int, N: int, n: int) -> Gate:
    x_qreg = QuantumRegister(2 * n, name='x')
    y_qreg = QuantumRegister(n, name='y')
    aux_qreg = QuantumRegister(n + 1, name='aux')

    circuit = QuantumCircuit(x_qreg,
                             y_qreg,
                             aux_qreg,
                             name=f'Exp({constant})_Mod_{N}')

    qft = QFT(n, do_swaps=False).to_gate()
    iqft = qft.inverse()

    for i in range(2 * n):
        partial_constant = pow(constant, pow(2, i), mod=N)
        circuit.append(
            controlled_modular_multiplication_gate(partial_constant, N, n, qft, iqft),
            list(chain([x_qreg[i]], y_qreg, aux_qreg))
        )

    return circuit.to_gate()


def controlled_modular_multiplication_gate(constant: int, N: int, n: int, qft: Gate, iqft: Gate) -> Gate:
    ctrl_qreg = QuantumRegister(1, name='ctrl')
    x_qreg = QuantumRegister(n, name='x')
    aux_qreg = QuantumRegister(n, name='aux')
    flag_qreg = QuantumRegister(1, name='flag')

    circuit = QuantumCircuit(ctrl_qreg,
                             x_qreg,
                             aux_qreg,
                             flag_qreg,
                             name=f'C-MM({constant})_Mod_{N}')

    circuit.append(
        _controlled_modular_product_sum_operator(constant, N, n, qft, iqft),
        chain.from_iterable(circuit.qregs)
    )

    for i in range(n):
        circuit.cswap(ctrl_qreg[0], x_qreg[i], aux_qreg[i])

    constant_inv = pow(constant, -1, mod=N)
    circuit.append(
        _controlled_modular_product_sum_operator(constant_inv, N, n, qft, iqft).inverse(),
        chain.from_iterable(circuit.qregs)
    )

    return circuit.to_gate()


def _controlled_modular_product_sum_operator(constant: int, N: int, n: int, qft: Gate, iqft: Gate) -> Gate:
    if n == 1:
        raise ValueError("Case n = 1 not supported")

    ctrl_qreg = QuantumRegister(1, name='ctrl')
    x_qreg = QuantumRegister(n, name='x')
    y_qreg = QuantumRegister(n, name='y')
    flag_qreg = QuantumRegister(1, name='flag')

    circuit = QuantumCircuit(ctrl_qreg,
                             x_qreg,
                             y_qreg,
                             flag_qreg,
                             name=f'CC-MPS_({constant})_Mod_{N}')

    for i in reversed(range(n)):
        partial_constant = (pow(2, i, mod=N) * constant) % N

        g_qreg = x_qreg[:]
        g_qreg.pop(i)

        circuit.append(
            _double_controlled_modular_adder(partial_constant, N, n, qft, iqft),
            chain(ctrl_qreg, [x_qreg[i]], y_qreg, g_qreg, flag_qreg)
        )

    return circuit.to_gate()


def _double_controlled_modular_adder(constant: int, N: int, n: int,
                                     qft: Gate,
                                     iqft: Gate) -> Gate:
    ctrl_qreg = QuantumRegister(2, name='ctrl')
    x_qreg = QuantumRegister(n, name='x')
    g_qreg = QuantumRegister(n - 1 if n >= 2 else 1, name='g')
    flag_qreg = QuantumRegister(1, name='flag')

    circuit = QuantumCircuit(ctrl_qreg,
                             x_qreg,
                             g_qreg,
                             flag_qreg,
                             name=f'CC-MA_({constant})_Mod_{N}')

    adder_regs = list(chain(flag_qreg, x_qreg))

    circuit.append(
        double_controlled_comparator(N - constant, n),
        circuit.qubits
    )

    circuit.append(qft, x_qreg)
    circuit.append(
        phi_constant_adder(get_angles(constant, n)).control(1),
        adder_regs
    )
    circuit.ccx(ctrl_qreg[0], ctrl_qreg[1], flag_qreg[0])
    circuit.append(
        phi_constant_adder(get_angles(N - constant, n)).control(1).inverse(),
        adder_regs
    )
    circuit.append(iqft, x_qreg)

    circuit.append(
        double_controlled_comparator(constant, n),
        circuit.qubits
    )

    return circuit.to_gate()
