from qiskit.circuit import Gate

from gates.haner.comparator import double_controlled_comparator, double_controlled_comparator_regs
from tests.haner.gate_experiments.comparator import ComparatorExperiment
from utils.typing_ import QRegsSpec


class Experiment(ComparatorExperiment):

    @property
    def _gate(self) -> Gate:
        return double_controlled_comparator(self._constant, self._n)

    @property
    def _qregs_spec(self) -> QRegsSpec:
        return double_controlled_comparator_regs(self._n)
