from itertools import chain
from typing import Callable, List

from qiskit import QuantumRegister
from qiskit.circuit import Gate, Qubit

from gates.haner.adder import adder, controlled_adder
from utils.circuit_creation import create_circuit
from utils.typing_ import QRegsSpec


def incrementer(n: int) -> Gate:
    return _incrementer(n, incrementer_regs, _subtractor, _substractor_qubits)


def incrementer_regs(n: int) -> QRegsSpec:
    return {
        'x': n,
        'g': n
    }


def _subtractor(n: int) -> Gate:
    return adder(n).inverse()


def _substractor_qubits(qregs: List[QuantumRegister]) -> List[Qubit]:
    x_qreg, g_qreg = qregs
    return list(chain(g_qreg, x_qreg))


def controlled_incrementer(n: int) -> Gate:
    return _incrementer(n, controlled_incrementer_regs, _controlled_subtractor, _controlled_substractor_qubits, 'C-')


def controlled_incrementer_regs(n: int) -> QRegsSpec:
    return {
        'ctrl': 1,
        **incrementer_regs(n)
    }


def _controlled_subtractor(n: int) -> Gate:
    return controlled_adder(n).inverse()


def _controlled_substractor_qubits(qregs: List[QuantumRegister]) -> List[Qubit]:
    ctrl_qreg, x_qreg, g_qreg = qregs
    return list(chain(ctrl_qreg, g_qreg, x_qreg))


def _incrementer(n: int,
                 regs_spec: Callable[[int], QRegsSpec],
                 gate: Callable[[int], Gate],
                 gate_qubits: Callable[[List[QuantumRegister]], List[Qubit]],
                 prefix: str = '') -> Gate:

    regs_spec = regs_spec(n)
    circuit = create_circuit(regs_spec, f'{prefix}Inc')

    keys = list(regs_spec.keys())
    g_qreg = circuit.qregs[keys.index('g')]

    for _ in range(2):
        circuit.append(
            gate(n),
            gate_qubits(circuit.qregs)
        )
        circuit.x(g_qreg)

    return circuit.to_gate()
