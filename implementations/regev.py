from typing import Union, Tuple, Optional

import numpy as np
from abc import ABC, abstractmethod
from itertools import chain

from qiskit import QuantumRegister, AncillaRegister, QuantumCircuit, ClassicalRegister

from qiskit.circuit import Instruction
from qiskit.circuit.library import QFT
from utils.circuit_creation import create_circuit
from utils.is_prime import is_prime
from utils.convert_measurement import convert_measurement

import logging
import math
from fractions import Fraction
from decimal import Decimal, getcontext

from qiskit.providers import  Backend
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

#from qiskit.utils.validation import validate_min

logger = logging.getLogger(__name__)
getcontext().prec = 1000

class Regev(ABC):

    def __init__(self,  shots) -> None:
        self.shots = shots
        self.result = RegevResult()

    def get_vector(self, N: int, d_ceil=False, qd_ceil=False, semi_classical=False) -> 'RegevResult':
        self._validate_input(N)


        circuit = self.construct_circuit(N, d_ceil, qd_ceil, semi_classical, measurement=True)
        aersim = AerSimulator()
        pm = generate_preset_pass_manager(backend=aersim, optimization_level=3)
        isa_qc = pm.run(circuit)

        counts = aersim.run(isa_qc, shots=self.shots).result().get_counts(0)
        # counts = result.get_counts(0)
        # print('Counts(ideal):', counts)

        # counts=self.sampler().run(circuit, shots=self.shots).result().quasi_dists[0].binary_probabilities()

        self.result.total_counts = len(counts)
        self.result.total_shots = self.shots
        print(f"counts.items(): {counts.items()}")

        sorted_counts_items = sorted(counts.items(), key=lambda x: x[1])

        for measurement, shots in sorted_counts_items:
            # measurement = self._parse_measurement(measurement, semi_classical)
            # print(f", measurment: {measurement}   |   shots: {shots}", end="")
            self.result.output_data.append([convert_measurement(measurement), measurement, shots])
            # order = self._get_order(measurement, a, N)
            # if order:
            #     if order == 1:
            #         logger.info('Skip trivial order.')
            #         continue
            #
            #     if result.order and not result.order == order:
            #         logger.error(f'Currently computed order {order} differs from already stored: {result.order}.')
            #         continue

                # result.order = order

            self.result.successful_counts += 1
            self.result.successful_shots += shots

        return self.result


    @staticmethod
    def _parse_measurement(measurement: str, semi_classical=False):
        if semi_classical:
            measurement = measurement.replace(' ', '')
        return int(measurement, base=2)


    def construct_circuit(self, N: int, d_ceil, qd_ceil, semi_classical: bool = False, measurement: bool = True):
        self._validate_input(N)

        n = N.bit_length()

        if d_ceil:
            d = math.ceil(math.sqrt(n))
        else:
            d = math.floor(math.sqrt(n))

        if qd_ceil:
            qd = math.ceil(n/d) + d
        else:
            qd = math.floor(n/d) + d

        # print(f"N: {N}\nn: {n}\nd: {d}\nqd: {qd}")
        self.result.N = N
        self.result.n = n
        self.result.d_ceil = d_ceil
        self.result.qd_ceil = qd_ceil
        self.result.number_of_primes = d
        self.result.exp_register_width = qd


        if semi_classical:
            if not measurement:
                raise ValueError('Semi-classical implementation have to contain measurement parts.')
            return self._construct_circuit_with_semiclassical_QFT(N, n, d, qd)
        else:
            return self._construct_circuit(N, n, measurement, d, qd)


    @staticmethod
    def generate_a(d: int, N: int):
        a = []
        ind = 0
        num = 2
        while ind < d:
            if is_prime(num):
                if N % num == 0:
                    print(f"We are very lucky! Here is p: {num} and q: {N//num}")
                    num += 1
                    continue
                a.append(int(math.pow(num, 2)))
                ind += 1
            num += 1
        return a


    @staticmethod
    def _validate_input(N: int):
        # validate_min('N', N, 3)
        # validate_min('a', a, 2)

        if N < 1 or N % 2 == 0:
            raise ValueError(f'The input N needs to be an odd integer greater than 1. Provided N = {N}.')


    def _construct_circuit(self, N: int, n: int, measurement: bool, d: int, qd: int) -> QuantumCircuit:


        # CZĘŚĆ BARTKA (utworzenie rejestrów)
        # x_qreg = QuantumRegister(2 * n, 'x')
        # y_qreg = QuantumRegister(n, 'y')
        # aux_qreg = AncillaRegister(self._get_aux_register_size(n), 'aux')
        # circuit = QuantumCircuit(x_qreg, y_qreg, aux_qreg, name=self._get_name(N, d))

        # CZĘŚĆ NiP (utworzenie rejestrów)

        x_qregs_spec = dict()
        a = self.generate_a(d, N)

        # Input registers, each has qd-qubits
        for i in range(d):
            x_qregs_spec[f'x{i + 1}'] = qd
        x_qregs = [QuantumRegister(size, name=name) for name, size in x_qregs_spec.items()]

        # Output register, has n qubits (because of mod N)
        y_qreg = QuantumRegister(n, 'y')

        # Creating quantum circuit
        aux_qreg = AncillaRegister(self._get_aux_register_size(n), 'aux')
        circuit = QuantumCircuit(*x_qregs, y_qreg, aux_qreg, name=self._get_name(N, d))

        # Initializing input register's qubits superposition
        for qreg in x_qregs:
            circuit.h(qreg)

        # Poniższa linijka była w kodzie Bartka, może się wiązać z jakimś bitem kontrolnym albo mnożeniem
        circuit.x(y_qreg[0])

        # Debugging
        # print(f"a: {a}\n\n")
        self.result.squared_primes = a
        # print(f"circuit.qubits: {circuit.qubits}")

        # for qubit in circuit.qubits:
        #     print(f"qubit: {qubit._repr}\nregister: {qubit._register}\nregister_name: {qubit._register._name}")
        # Koniec debuggingu

        x_regs_cubits = []
        # for qubit in circuit.qubits:
        qregs_all = circuit.qregs
        # print(f"\n\nqregs_all: {qregs_all}")
        # print(f"qregs_all[0].qubits: {qregs_all[0]._bits}")

        for i in range(d):
            qubits_to_pass = []
            qubits_to_pass += qregs_all[i]
            qubits_to_pass += qregs_all[-2]
            qubits_to_pass += qregs_all[-1]
            # print(f"\nqubits_to_pass: {qubits_to_pass}\na_to_pass: {a[i]}\n")

            modular_exponentiation_gate = self._modular_exponentiation_gate(a[i], N, n, qd)
            circuit.append(
                modular_exponentiation_gate,
                qubits_to_pass
            )

        qft = QFT(qd).to_gate()


        # UWAGA! TEN FRAGMENT KODU ZAKOMENTOWANY NA POTRZEBY TESTU, NORMALNIE MUSI BYĆ ODKOMENTOWANY

        for i in range(d):
            circuit.append(
                qft,
                qregs_all[i]
        )

        if measurement:
            for i in range(d):
                x_creg = ClassicalRegister(qd, name=f'x{i+1}Value')
                circuit.add_register(x_creg)
                circuit.measure(qregs_all[i], x_creg)
            # y_creg = ClassicalRegister(n, 'yValue')
            # circuit.add_register(y_creg)
            # circuit.measure(qregs_all[-2], y_creg)

        return circuit


    def _construct_circuit_with_semiclassical_QFT(self, N: int, n: int, d: int, qd: int) -> QuantumCircuit:

        # CZĘŚĆ NiP (utworzenie rejestrów)

        x_qregs_spec = dict()
        a = self.generate_a(d, N)

        # Input registers, each has qd-qubits
        for i in range(d):
            x_qregs_spec[f'x{i + 1}'] = qd
        x_qregs = [QuantumRegister(size, name=name) for name, size in x_qregs_spec.items()]

        # Output register, has n qubits (because of mod N)
        y_qreg = QuantumRegister(n, 'y')

        # Creating quantum circuit
        aux_qreg = AncillaRegister(self._get_aux_register_size(n), 'aux')
        circuit = QuantumCircuit(*x_qregs, y_qreg, aux_qreg, name=self._get_name(N, d))



        x_qreg = QuantumRegister(1, 'x')
        y_qreg = QuantumRegister(n, 'y')
        aux_qreg = AncillaRegister(self._get_aux_register_size(n), 'aux')

        x_creg = [ClassicalRegister(1, f'xV{i}') for i in range(2 * n)]
        aux_creg = ClassicalRegister(1, 'auxValue')

        name = f'{self._get_name(a, N)} (semi-classical QFT)'
        circuit = QuantumCircuit(x_qreg, y_qreg, aux_qreg, *x_creg, aux_creg, name=name)

        circuit.x(y_qreg[0])

        max_i = 2 * n - 1
        for i in range(0, 2 * n):
            circuit.h(x_qreg)

            partial_constant = pow(a, pow(2, max_i - i), mod=N)
            modular_multiplication_gate = self._modular_multiplication_gate(partial_constant, N, n)
            circuit.append(
                modular_multiplication_gate,
                chain([x_qreg[0]], y_qreg, aux_qreg)
            )

            for j in range(i):
                angle = -np.pi / float(pow(2, i - j))
                circuit.p(angle, x_qreg[0]).c_if(x_creg[j], 1)

            circuit.h(x_qreg)
            circuit.measure(x_qreg[0], x_creg[i][0])
            circuit.measure(x_qreg[0], aux_creg[0])
            circuit.x(x_qreg).c_if(aux_creg, 1)

        circuit.measure(x_qreg[0], aux_creg[0])

        return circuit


    @abstractmethod
    def _get_aux_register_size(self, n: int) -> int:
        raise NotImplemented

    def _get_name(self, N: int, d: int) -> str:
        return f'{self._prefix} Regev(N={N}, d={d})'

    @property
    @abstractmethod
    def _prefix(self) -> str:
        raise NotImplemented

    @abstractmethod
    def _modular_exponentiation_gate(self, constant: int, N: int, n: int, qd: int) -> Instruction:
        raise NotImplemented

    @abstractmethod
    def _modular_multiplication_gate(self, constant: int, N: int, n: int) -> Instruction:
        raise NotImplemented


    def get_factors(self, vect, t_a, t_N):
        a = self.result.squared_primes
        N = self.result.N
        a = t_a
        N = t_N
        prod = Decimal(1)
        for i in range(len(a)):
            sqrt_a = Decimal(a[i]).sqrt()
            pow_a = ((sqrt_a ** vect[i]) % N)
            prod = ((prod*pow_a) % N)

        print(f"prod: {prod}")
        # prod = (prod % Decimal(N))

        val1 = (prod - 1)
        val2 = (prod + 1)
        print(f"val1: {val1}\nval2: {val2}\nN: {N}")

        p = math.gcd(int(val2), N)

        if p == N:
            print(f"We've got bad luck number one - p and q are both dividers of ({val2} + 1)")
            return -1
        elif p == 1:
            print(f"We've got bad luck number two - p and q are both dividers of ({val2} - 1)")
            return -1

        q = int(N/p)

        return p, q


class RegevResult:

    def __init__(self) -> None:
        self._order = None
        self._total_counts = 0
        self._successful_counts = 0
        self._total_shots = 0
        self._successful_shots = 0

        self._N = 0
        self._n = 0
        self._d_ceil = False
        self._qd_ceil = False
        self._number_of_primes = 0
        self._exp_register_width = 0
        self._squared_primes = []
        self._output_data = []


    @property
    def order(self) -> Optional[int]:
        return self._order

    @order.setter
    def order(self, value: int) -> None:
        self._order = value

    @property
    def total_counts(self) -> int:
        return self._total_counts

    @total_counts.setter
    def total_counts(self, value: int) -> None:
        self._total_counts = value

    @property
    def successful_counts(self) -> int:
        return self._successful_counts

    @successful_counts.setter
    def successful_counts(self, value: int) -> None:
        self._successful_counts = value

    @property
    def total_shots(self) -> int:
        return self._total_shots

    @total_shots.setter
    def total_shots(self, value: int) -> None:
        self._total_shots = value

    @property
    def successful_shots(self) -> int:
        return self._successful_shots

    @successful_shots.setter
    def successful_shots(self, value: int) -> None:
        self._successful_shots = value



    @property
    def N(self) -> int:
        return self._N

    @N.setter
    def N(self, value: int) -> None:
        self._N = value

    @property
    def n(self) -> int:
        return self._n

    @n.setter
    def n(self, value: int) -> None:
        self._n = value

    @property
    def d_ceil(self) -> bool:
        return self._d_ceil

    @d_ceil.setter
    def d_ceil(self, value: bool) -> None:
        self._d_ceil = value

    @property
    def qd_ceil(self) -> bool:
        return self._qd_ceil

    @qd_ceil.setter
    def qd_ceil(self, value: bool) -> None:
        self._qd_ceil = value

    @property
    def number_of_primes(self) -> int:
        return self._number_of_primes

    @number_of_primes.setter
    def number_of_primes(self, value: int) -> None:
        self._number_of_primes = value

    @property
    def exp_register_width(self) -> int:
        return self._exp_register_width

    @exp_register_width.setter
    def exp_register_width(self, value: int) -> None:
        self._exp_register_width = value

    @property
    def squared_primes(self) -> []:
        return self._squared_primes

    @squared_primes.setter
    def squared_primes(self, value: []) -> None:
        self._squared_primes = value

    @property
    def output_data(self) -> []:
        return self._output_data

    @output_data.setter
    def output_data(self, value: []) -> None:
        self._output_data = value




