from qiskit.circuit import Gate

from gates.haner.incrementer import controlled_incrementer, controlled_incrementer_regs
from tests.haner.gate_experiments.incrementer import IncrementerExperiment
from utils.typing_ import QRegsSpec


class Experiment(IncrementerExperiment):

    @property
    def _gate(self) -> Gate:
        return controlled_incrementer(self._n)

    @property
    def _qregs_spec(self) -> QRegsSpec:
        return controlled_incrementer_regs(self._n)
