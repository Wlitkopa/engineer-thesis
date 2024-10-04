from qiskit.circuit import Gate

from gates.haner.adder import controlled_adder, controlled_adder_regs
from tests.haner.gate_experiments.adder import AdderExperiment
from utils.typing_ import QRegsSpec


class Experiment(AdderExperiment):

    @property
    def _gate(self) -> Gate:
        return controlled_adder(self._n)

    @property
    def _qregs_spec(self) -> QRegsSpec:
        return controlled_adder_regs(self._n)
