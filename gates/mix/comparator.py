from itertools import chain

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Gate

from gates.beauregard.constant_adder import as_bits_reversed


def double_controlled_comparator(constant: int, n: int) -> Gate:
    ctrl_qreg = QuantumRegister(2, name='ctrl')
    x_qreg = QuantumRegister(n, name='x')
    g_qreg = QuantumRegister(n - 1 if n >= 2 else 1, name='g')
    c_qreg = QuantumRegister(1, name='c')

    circuit = QuantumCircuit(ctrl_qreg,
                             x_qreg,
                             g_qreg,
                             c_qreg,
                             name=f'CC-CMP_({constant})')

    cccx = _triple_controlled_not()
    cccx_qubits = list(chain(ctrl_qreg, x_qreg, c_qreg, g_qreg)) if n == 1 \
        else list(chain(ctrl_qreg, [g_qreg[n - 2]], c_qreg, [x_qreg[0]]))

    circuit.x(x_qreg)

    if n == 1:
        if constant == 1:
            circuit.append(cccx, cccx_qubits)
    else:
        body = _carry_body(constant, n, x_qreg, g_qreg)
        body_qubits = list(chain(x_qreg, g_qreg))

        circuit.append(cccx, cccx_qubits)
        circuit.append(body, body_qubits)
        circuit.append(cccx, cccx_qubits)
        circuit.append(body.inverse(), body_qubits)

    circuit.x(x_qreg)

    return circuit.to_gate()


def _carry_body(constant: int, n: int, x_qreg: QuantumRegister, g_qreg: QuantumRegister) -> Gate:
    constant_bits = as_bits_reversed(constant, n)
    circuit = QuantumCircuit(x_qreg, g_qreg)

    for i in reversed(range(2, n)):
        if constant_bits[i] == '1':
            circuit.cx(x_qreg[i], g_qreg[i - 1])
            circuit.x(x_qreg[i])
        circuit.ccx(g_qreg[i - 2], x_qreg[i], g_qreg[i - 1])

    if constant_bits[1] == '1':
        circuit.cx(x_qreg[1], g_qreg[0])
        circuit.x(x_qreg[1])

    if constant_bits[0] == '1':
        circuit.ccx(x_qreg[0], x_qreg[1], g_qreg[0])

    for i in range(2, n):
        circuit.ccx(g_qreg[i - 2], x_qreg[i], g_qreg[i - 1])

    return circuit.to_gate()


def _triple_controlled_not() -> Gate:
    ctrl_qreg = QuantumRegister(3, name='ctrl')
    x_qreg = QuantumRegister(1, name='x')
    g_qreg = QuantumRegister(1, name='g')

    circuit = QuantumCircuit(ctrl_qreg,
                             x_qreg,
                             g_qreg,
                             name='CCCX')

    for _ in range(2):
        circuit.ccx(ctrl_qreg[2], g_qreg[0], x_qreg[0])
        circuit.ccx(ctrl_qreg[0], ctrl_qreg[1], g_qreg[0])

    return circuit.to_gate()
