from itertools import chain

from qiskit.circuit import Gate

from gates.haner.carry import controlled_carry
from gates.haner.incrementer import controlled_incrementer
from utils.circuit_creation import create_circuit
from utils.typing_ import QRegsSpec


def controlled_constant_adder(constant: int, n: int) -> Gate:
    circuit = create_circuit(controlled_constant_adder_regs(n), f'C-Add_({constant})')
    ctrl_qreg, x_qreg, g_qreg = circuit.qregs

    if n == 1:
        if constant == 1:
            circuit.cx(ctrl_qreg[0], x_qreg[0])
    else:
        mid = n // 2 + n % 2

        low_part = constant & ((1 << mid) - 1)
        high_part = constant >> mid

        low_qreg = x_qreg[:mid]
        high_qreg = x_qreg[mid:]

        carry = controlled_carry(low_part, mid)
        inc = controlled_incrementer(n - mid)

        inc_regs = list(chain(g_qreg, high_qreg, low_qreg[:len(high_qreg)]))
        carry_regs = list(chain(ctrl_qreg, low_qreg, high_qreg[:(mid - 1)], g_qreg))

        circuit.append(inc, inc_regs)

        for i in range(len(high_qreg)):
            circuit.cx(g_qreg[0], high_qreg[i])

        circuit.append(carry, carry_regs)
        circuit.append(inc, inc_regs)
        circuit.append(carry, carry_regs)

        for i in range(len(high_qreg)):
            circuit.cx(g_qreg[0], high_qreg[i])

        circuit.append(controlled_constant_adder(low_part, mid), chain(ctrl_qreg, low_qreg, g_qreg))
        circuit.append(controlled_constant_adder(high_part, n - mid), chain(ctrl_qreg, high_qreg, g_qreg))

    return circuit.to_gate()


def controlled_constant_adder_regs(n: int) -> QRegsSpec:
    return {
        'ctrl': 1,
        'x': n,
        'g': 1
    }
