from utils.circuit_creation import create_circuit
from utils.typing_ import QRegsSpec


def triple_controlled_not():
    circuit = create_circuit(cccx_regs(), 'CCCX')
    ctrl_qreg, x_qreg, g_qreg = circuit.qregs

    for _ in range(2):
        circuit.ccx(ctrl_qreg[2], g_qreg[0], x_qreg[0])
        circuit.ccx(ctrl_qreg[0], ctrl_qreg[1], g_qreg[0])

    return circuit.to_gate()


def cccx_regs() -> QRegsSpec:
    """ definition of CCCX registers
    |c> - control register
    |x> - target register
    |g> - dirty ancillary register; initial state must be restored
    """
    return {
        'ctrl': 3,
        'x': 1,
        'g': 1
    }
