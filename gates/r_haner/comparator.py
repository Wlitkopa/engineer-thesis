from itertools import chain
from typing import Callable

from qiskit.circuit import Gate

from gates.haner.carry import carry, carry_regs, controlled_carry, controlled_carry_regs, double_controlled_carry, \
    double_controlled_carry_regs
from utils.circuit_creation import create_circuit
from utils.typing_ import QRegsSpec


def comparator(constant: int, n: int) -> Gate:
    return _comparator(constant, n, carry_regs, carry)


def comparator_regs(n: int) -> QRegsSpec:
    return carry_regs(n)


def controlled_comparator(constant: int, n: int) -> Gate:
    return _comparator(constant, n, controlled_carry_regs, controlled_carry, 'C-')


def controlled_comparator_regs(n: int) -> QRegsSpec:
    return controlled_carry_regs(n)


def double_controlled_comparator(constant: int, n: int) -> Gate:
    return _comparator(constant, n, double_controlled_carry_regs, double_controlled_carry, 'CC-')


def double_controlled_comparator_regs(n: int) -> QRegsSpec:
    return double_controlled_carry_regs(n)


def _comparator(constant: int,
                n: int,
                regs_spec: Callable[[int], QRegsSpec],
                gate: Callable[[int, int], Gate],
                prefix: str = '') -> Gate:

    regs_spec = regs_spec(n)
    circuit = create_circuit(regs_spec, f'{prefix}Comp_({constant})')

    keys = list(regs_spec.keys())
    x_qreg = circuit.qregs[keys.index('x')]

    circuit.x(x_qreg)
    circuit.append(
        gate(constant, n),
        chain.from_iterable(circuit.qregs)
    )
    circuit.x(x_qreg)

    return circuit.to_gate()
