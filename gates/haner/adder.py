from typing import Callable, List

from qiskit import QuantumRegister
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import CXGate, CCXGate

from utils.circuit_creation import create_circuit
from utils.typing_ import QRegsSpec


def adder(n: int) -> Gate:
    return _adder(n, adder_regs, CXGate, _cx_qubits)


def _cx_qubits(qregs: List[QuantumRegister], i: int) -> List[Qubit]:
    x_qreg, y_qreg = qregs
    return [x_qreg[i], y_qreg[i]]


def adder_regs(n: int) -> QRegsSpec:
    return {
        'x': n,
        'y': n
    }


def controlled_adder(n: int) -> Gate:
    return _adder(n, controlled_adder_regs, CCXGate, _ccx_qubits, 'C-')


def controlled_adder_regs(n: int) -> QRegsSpec:
    return {
        'ctrl': 1,
        **adder_regs(n)
    }


def _ccx_qubits(qregs: List[QuantumRegister], i: int) -> List[Qubit]:
    ctrl_qreg, x_qreg, y_qreg = qregs
    return [ctrl_qreg[0], x_qreg[i], y_qreg[i]]


def _adder(n: int,
           regs_spec: Callable[[int], QRegsSpec],
           gate: Callable[[], Gate],
           gate_qubits: Callable[[List[QuantumRegister], int], List[Qubit]],
           prefix: str = '') -> Gate:

    regs_spec = regs_spec(n)
    circuit = create_circuit(regs_spec, f'{prefix}Adder')
    gate = gate()

    keys = list(regs_spec.keys())
    x_qreg = circuit.qregs[keys.index('x')]
    y_qreg = circuit.qregs[keys.index('y')]

    for i in range(1, n):
        circuit.cx(x_qreg[i], y_qreg[i])

    for i in reversed(range(1, n-1)):
        circuit.cx(x_qreg[i], x_qreg[i+1])

    for i in range(0, n-1):
        circuit.ccx(x_qreg[i], y_qreg[i], x_qreg[i+1])

    for i in reversed(range(1, n)):
        circuit.append(gate, gate_qubits(circuit.qregs, i))
        circuit.ccx(x_qreg[i-1], y_qreg[i-1], x_qreg[i])

    for i in range(1, n-1):
        circuit.cx(x_qreg[i], x_qreg[i+1])

    circuit.append(gate, gate_qubits(circuit.qregs, 0))
    for i in range(1, n):
        circuit.cx(x_qreg[i], y_qreg[i])

    return circuit.to_gate()
