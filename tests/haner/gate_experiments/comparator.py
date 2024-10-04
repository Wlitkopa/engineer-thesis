from abc import ABCMeta

from qiskit.circuit import Gate

from gates.haner.comparator import comparator, comparator_regs
from tests.haner.gate_experiments.carry import CarryExperiment
from utils.typing_ import QRegsSpec


class ComparatorExperiment(CarryExperiment, metaclass=ABCMeta):
    pass


class Experiment(ComparatorExperiment):

    @property
    def _gate(self) -> Gate:
        return comparator(self._constant, self._n)

    @property
    def _qregs_spec(self) -> QRegsSpec:
        return comparator_regs(self._n)
