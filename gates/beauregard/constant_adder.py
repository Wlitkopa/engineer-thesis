from typing import Union

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Gate


def phi_constant_adder(angles: Union[np.ndarray, ParameterVector]) -> Gate:
    circuit = QuantumCircuit(len(angles), name="phi_add_a")
    for i, angle in enumerate(angles):
        circuit.p(angle, i)
    return circuit.to_gate()


def get_angles(constant: int, n: int) -> np.ndarray:
    constant_bits = as_bits_reversed(constant, n)

    angles = np.zeros(n)
    for i in range(n):
        for j in range(i + 1):
            k = i - j
            if constant_bits[j] == '1':
                angles[i] += pow(2, -k)

    return angles * np.pi


def as_bits_reversed(constant: int, n: int):
    return (bin(int(constant))[2:].zfill(n))[::-1]
