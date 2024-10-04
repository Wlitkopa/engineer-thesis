from qiskit.circuit import Gate

from gates.haner.modular_exponentiation import modular_exponentiation_gate, modular_exponentiation_gate_regs
from tests.gate_experiment import GateExperiment
from utils.typing_ import QRegsSpec


class Experiment(GateExperiment):
    def __init__(self, constant: int, N: int, n: int):
        super().__init__()
        self._constant = constant
        self._N = N
        self._n = n

    @property
    def _gate(self) -> Gate:
        return modular_exponentiation_gate(self._constant, self._N, self._n)

    @property
    def _qregs_spec(self) -> QRegsSpec:
        return modular_exponentiation_gate_regs(self._n)
