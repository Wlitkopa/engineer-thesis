from qiskit.circuit import Gate

from gates.haner.carry import controlled_carry, controlled_carry_regs
from tests.haner.gate_experiments.carry import CarryExperiment
from utils.typing_ import QRegsSpec


class Experiment(CarryExperiment):

    @property
    def _gate(self) -> Gate:
        return controlled_carry(self._constant, self._n)

    @property
    def _qregs_spec(self) -> QRegsSpec:
        return controlled_carry_regs(self._n)
