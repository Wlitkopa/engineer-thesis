from qiskit.circuit import Gate

from gates.haner.comparator import controlled_comparator, controlled_comparator_regs
from tests.haner.gate_experiments.comparator import ComparatorExperiment
from utils.typing_ import QRegsSpec


class Experiment(ComparatorExperiment):

    @property
    def _gate(self) -> Gate:
        return controlled_comparator(self._constant, self._n)

    @property
    def _qregs_spec(self) -> QRegsSpec:
        return controlled_comparator_regs(self._n)
