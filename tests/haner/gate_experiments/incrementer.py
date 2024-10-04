from abc import ABCMeta

from qiskit.circuit import Gate

from gates.haner.incrementer import incrementer, incrementer_regs
from tests.gate_experiment import GateExperiment
from utils.typing_ import QRegsSpec


class IncrementerExperiment(GateExperiment, metaclass=ABCMeta):
    def __init__(self, n: int):
        super().__init__()
        self._n = n


class Experiment(IncrementerExperiment):

    @property
    def _gate(self) -> Gate:
        return incrementer(self._n)

    @property
    def _qregs_spec(self) -> QRegsSpec:
        return incrementer_regs(self._n)
