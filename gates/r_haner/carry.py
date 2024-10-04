from itertools import chain
from typing import Callable, List

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import CXGate, CCXGate

from gates.haner.cccx import triple_controlled_not
from utils.bits import as_bits_reversed
from utils.circuit_creation import create_circuit
from utils.typing_ import QRegsSpec


def carry(constant: int, n: int) -> Gate:
    return _carry(constant, n, carry_regs, CXGate, _cx_qubits)


def _cx_qubits(qregs: List[QuantumRegister], n: int) -> List[Qubit]:
    if n == 1:
        return list(chain.from_iterable(qregs))
    else:
        _, g_qreg, c_qreg = qregs
        return [g_qreg[n-2], c_qreg[0]]


def carry_regs(n: int) -> QRegsSpec:
    """ return definition of carry gate registers
    |x> - input register
    |g> - dirty ancillary register; initial state must be restored
    |c> - output register - information about carry
    """
    if n == 1:
        return {
            'x': n,
            'c': 1
        }
    else:
        return {
            'x': n,
            'g': n - 1,
            'c': 1
        }


def controlled_carry(constant: int, n: int) -> Gate:
    return _carry(constant, n, controlled_carry_regs, CCXGate, _ccx_qubits, 'C-')


def _ccx_qubits(qregs: List[QuantumRegister], n: int) -> List[Qubit]:
    if n == 1:
        return list(chain.from_iterable(qregs))
    else:
        ctrl_qreg, _, g_qreg, c_qreg = qregs
        return [ctrl_qreg[0], g_qreg[n - 2], c_qreg[0]]


def controlled_carry_regs(n: int) -> QRegsSpec:
    """ return definition of carry gate registers
    |ctrl> - control register
    |x> - input register
    |g> - dirty ancillary register; initial state must be restored
    |c> - output register - information about carry
    """
    return {
        'ctrl': 1,
        **carry_regs(n)
    }


def double_controlled_carry(constant: int, n: int) -> Gate:
    return _carry(constant, n, double_controlled_carry_regs, triple_controlled_not, _cccx_qubits, 'CC-')


def _cccx_qubits(qregs: List[QuantumRegister], n: int) -> List[Qubit]:
    ctrl_qreg, x_qreg, g_qreg, c_qreg = qregs
    if n == 1:
        return list(chain(ctrl_qreg, x_qreg, c_qreg, g_qreg))
    else:
        return list(chain(ctrl_qreg, [g_qreg[n - 2]], c_qreg, [x_qreg[0]]))


def double_controlled_carry_regs(n: int) -> QRegsSpec:
    """ return definition of carry gate registers
    |ctrl> - control register
    |x> - input register
    |g> - dirty ancillary register; initial state must be restored
    |c> - output register - information about carry
    """
    return {
        'ctrl': 2,
        'x': n,
        'g': n - 1 if n >= 2 else 1,
        'c': 1
    }


def _carry(constant: int,
           n: int,
           regs_spec: Callable[[int], QRegsSpec],
           gate: Callable[[], Gate],
           gate_qubits: Callable[[List[QuantumRegister], int], List[Qubit]],
           prefix: str = '') -> Gate:

    regs_spec = regs_spec(n)
    circuit = create_circuit(regs_spec, f'{prefix}Carry_({constant})')
    gate = gate()
    gate_qubits = gate_qubits(circuit.qregs, n)

    if n == 1:
        if constant == 1:
            circuit.append(gate, gate_qubits)
    else:
        keys = list(regs_spec.keys())
        x_qreg = circuit.qregs[keys.index('x')]
        g_qreg = circuit.qregs[keys.index('g')]
        body = _carry_body(constant, n, x_qreg, g_qreg)
        body_qubits = list(chain(x_qreg, g_qreg))

        circuit.append(gate, gate_qubits)
        circuit.append(body, body_qubits)
        circuit.append(gate, gate_qubits)
        circuit.append(body.inverse(), body_qubits)

    return circuit.to_gate()


def _carry_body(constant: int, n: int, x_qreg: QuantumRegister, g_qreg: QuantumRegister) -> Gate:
    circuit = QuantumCircuit(x_qreg, g_qreg, name=f'Carry_({constant})_body')
    constant_bits = as_bits_reversed(constant, n)

    for i in reversed(range(2, n)):
        if constant_bits[i] == '1':
            circuit.cx(x_qreg[i], g_qreg[i-1])
            circuit.x(x_qreg[i])
        circuit.ccx(g_qreg[i-2], x_qreg[i], g_qreg[i-1])

    if constant_bits[1] == '1':
        circuit.cx(x_qreg[1], g_qreg[0])
        circuit.x(x_qreg[1])

    if constant_bits[0] == '1':
        circuit.ccx(x_qreg[0], x_qreg[1], g_qreg[0])

    for i in range(2, n):
        circuit.ccx(g_qreg[i-2], x_qreg[i], g_qreg[i-1])

    return circuit.to_gate()
