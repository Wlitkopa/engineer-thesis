from abc import ABCMeta

from qiskit.circuit import Gate

from gates.haner.carry import carry, carry_regs
from tests.gate_experiment import GateExperiment
from utils.typing_ import QRegsSpec


class CarryExperiment(GateExperiment, metaclass=ABCMeta):
    def __init__(self, constant: int, n: int):
        super().__init__()
        self._constant = constant
        self._n = n


class Experiment(CarryExperiment):

    @property
    def _gate(self) -> Gate:
        return carry(self._constant, self._n)

    @property
    def _qregs_spec(self) -> QRegsSpec:
        return carry_regs(self._n)
