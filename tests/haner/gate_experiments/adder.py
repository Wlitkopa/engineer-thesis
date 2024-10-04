from abc import ABCMeta

from qiskit.circuit import Gate

from gates.haner.adder import adder, adder_regs
from tests.gate_experiment import GateExperiment
from utils.typing_ import QRegsSpec


class AdderExperiment(GateExperiment, metaclass=ABCMeta):
    def __init__(self, n: int):
        super().__init__()
        self._n = n


class Experiment(AdderExperiment):

    @property
    def _gate(self) -> Gate:
        return adder(self._n)

    @property
    def _qregs_spec(self) -> QRegsSpec:
        return adder_regs(self._n)
