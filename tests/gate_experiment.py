from abc import ABC, abstractmethod
from itertools import chain
from typing import List

# from qiskit import QuantumCircuit, ClassicalRegister, execute, Aer
from qiskit.circuit import Gate
from qiskit.providers import Backend
from qiskit.result import Result

from utils.bits import as_bits_reversed
from utils.circuit_creation import create_circuit
from utils.typing_ import QRegsSpec, Name, ValuesMap

# Moje modyfikacje

# Zamiast linijki "from qiskit import QuantumCircuit, ClassicalRegister, execute, Aer"
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import Aer
from qiskit.primitives import BackendSampler
from qiskit.providers.basic_provider import BasicSimulator




class GateExperiment(ABC):

    def __init__(self, backend: Backend = Aer.get_backend('qasm_simulator')):
        self._backend = backend

    # def run(self, initial_values: ValuesMap, name: Name) -> Result:
    #     circuit = self._construct_circuit(initial_values, name)
    #     job = execute(circuit, self._backend, shots=1)
    #     return job.result()

    # Zamiast metody "__init__"
    # def __init__(self, backend: Backend = BasicSimulator()):
    #     self._backend = backend

    # Zamiast metody "run"
    def run(self, initial_values: ValuesMap, name: Name) -> Result:
        # backend = BasicSimulator()
        # backend = Aer.get_backend('qasm_simulator')

        circuit = self._construct_circuit(initial_values, name)
        new_circuit = transpile(circuit, self._backend)
        job = self._backend.run(new_circuit)
        return job.result()

    def _construct_circuit(self, initial_values: ValuesMap, name: Name) -> QuantumCircuit:
        circuit = create_circuit(self._qregs_spec, name)
        _initialize(circuit, initial_values)
        _append_gate(circuit, self._gate)
        _measure(circuit)
        return circuit

    @property
    @abstractmethod
    def _gate(self) -> Gate:
        pass

    @property
    @abstractmethod
    def _qregs_spec(self) -> QRegsSpec:
        pass

    @property
    def qreg_names(self) -> List[Name]:
        return [name for name, _ in self._qregs_spec.items()]


def _initialize(circuit: QuantumCircuit, values: ValuesMap) -> None:
    qregs = circuit.qregs
    for qreg in qregs:
        value = values.get(qreg.name, 0)
        value_str = as_bits_reversed(value, qreg.size)

        for i, bit in enumerate(value_str):
            if bit == '1':
                circuit.x(qreg[i])

    circuit.barrier(*qregs)


def _append_gate(circuit: QuantumCircuit, gate: Gate) -> None:
    qubits = list(chain.from_iterable(circuit.qregs))
    circuit.append(gate, qubits)


def _measure(circuit: QuantumCircuit) -> None:
    qregs = circuit.qregs
    cregs = [ClassicalRegister(qreg.size, name=f'{qreg.name}Value') for qreg in qregs]
    circuit.add_register(*cregs)

    circuit.barrier(*qregs)

    for qreg, creg in zip(qregs, cregs):
        circuit.measure(qreg[:], creg[:])
