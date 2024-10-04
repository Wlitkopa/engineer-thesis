from qiskit.circuit import Gate

from gates.haner.cccx import triple_controlled_not, cccx_regs
from tests.gate_experiment import GateExperiment
from utils.typing_ import QRegsSpec


class Experiment(GateExperiment):

    @property
    def _gate(self) -> Gate:
        return triple_controlled_not()

    @property
    def _qregs_spec(self) -> QRegsSpec:
        return cccx_regs()
