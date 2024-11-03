from typing import Union, Tuple, Optional

import numpy as np
from abc import ABC, abstractmethod
from itertools import chain, combinations

from qiskit import QuantumRegister, AncillaRegister, QuantumCircuit, ClassicalRegister

from qiskit.circuit import Instruction
from qiskit.circuit.library import QFT
from utils.circuit_creation import create_circuit
from utils.is_prime import is_prime
from utils.convert_measurement import convert_measurement
from utils.convert_to_matrix_row import convert_to_matrix_row
from utils.convert_milliseconds import convert_milliseconds

import logging
import math
import olll
from random import shuffle
from fractions import Fraction
from decimal import Decimal, getcontext
import time


# Importy z data_analizer.py
import os
import ast
import math
import olll
import itertools
import numpy as np


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
        self.vectors = []


    def run_all_algorithm(self):
        pass

    def run_classical_part(self):
        pass

    def run_quantum_part(self, Ns, d_qd_list):

        for i in range(len(d_qd_list)):
            d_ceil_bool = d_qd_list[i][0]
            qd_ceil_bool = d_qd_list[i][1]

            for j in range(len(Ns)):

                N = Ns[j]
                print(f"\nN: {N}")

                start = time.time()
                result = self.get_vectors(N, d_ceil=d_ceil_bool, qd_ceil=qd_ceil_bool, semi_classical=False)
                end = time.time()
                exec_time = (end - start) * (10 ** 3)
                converted_time = convert_milliseconds(exec_time)

                vectors = convert_to_matrix_row(result.output_data)

                result_str = (f"N: {result.N}\n"
                              f"n: {result.n}\n"
                              f"d_ceil: {result.d_ceil}\n"
                              f"qd_ceil: {result.qd_ceil}\n"
                              f"number_of_primes (d): {result.number_of_primes}\n"
                              f"exp_register_width (qd): {result.exp_register_width}\n"
                              f"squared_primes: {result.squared_primes}\n"
                              f"output_data: {result.output_data}\n"
                              f"\nvectors: {vectors}\n"
                              f"\nexec_time (ms): {exec_time} ms\n"
                              f"exec_time: {converted_time}")

                if d_ceil_bool:
                    d_mode = "ceil"
                else:
                    d_mode = "floor"

                if qd_ceil_bool:
                    qd_mode = "ceil"
                else:
                    qd_mode = "floor"

                file = open(f"output_data/regev/quantum_part_2/{d_mode}_{qd_mode}/N_{N}", "w")
                file.write(result_str)
                file.close()

                print(f"N: {result.N}")
                print(f"n: {result.n}")
                print(f"d_ceil: {result.d_ceil}")
                print(f"qd_ceil: {result.qd_ceil}")
                print(f"number_of_primes: {result.number_of_primes}")
                print(f"exp_register_width: {result.exp_register_width}")
                print(f"squared_primes: {result.squared_primes}")
                print(f"output_data: {result.output_data}")
                print(f"\nvectors: {vectors}\n")
                print(f"exec_time: {exec_time}ms")
                print(f"converted_time: {converted_time}")


    def run_file_data_analyzer(self, file_name):

        if not os.path.exists(file_name):
            print(f"File {file_name} doesn't exists")
            return -1

        start = time.time()

        result = ""
        vectors = []
        p_q_vectors = []

        dir1_part = file_name.split("/")[-2].split("_")[0]
        dir2_part = file_name.split("/")[-2].split("_")[1]

        print(f"dir1_part: {dir1_part}\ndir2_part: {dir2_part}")

        with open(file_name) as results:

            dq = 0
            for i in range(10):
                line = results.readline()
                if i == 0:
                    N = int(line.split(' ')[1])
                if i == 4:
                    d = int(line.split(':')[1][:-1])
                if i == 5:
                    dq = int(line.split(':')[1][:-1])
                if i == 6:
                    a = ast.literal_eval(line.split(':')[1])
                    a_root = []
                    for a_ in a:
                        a_root.append(int(math.sqrt(a_)))

            while (line := results.readline()) != '\n':
                v = line.split(':')[1][:-2]
                duplicate = int(line.split(' ')[2])
                # for i in range(min(d+4, duplicate)):
                vectors.append(ast.literal_eval(v))

            shuffle(vectors)
            print(vectors)
            print(dq)

        qd = dq

        n = N.bit_length()
        result += (f"N: {N}\n"
                   f"n: {n}\n"
                   f"number_of_primes (d): {d}\n"
                   f"exp_register_width (qd): {qd}\n"
                   f"squared primes (a): {a_root}\n\n")


        # self.get_vectors(N, d_ceil, qd_ceil, semi_classical)
        np.set_printoptions(suppress=True)
        # if len(self.vectors) == 0:
        #     return -1
        #
        # d = self.result.number_of_primes
        # qd = self.result.exp_register_width
        # a = self.result.squared_primes
        # vectors = self.vectors

        T = 2
        R = math.ceil(6 * T * math.sqrt((d + 5) * (2 * d) + 4) * (d / 2) * (2 ** ((qd + 1) / (d + 4) + d + 2)))
        t = 1 + math.ceil(math.log(math.sqrt(d) * R, 2))
        delta = math.sqrt(d / 2) / R
        delta_inv = R / math.sqrt(d / 2)

        result += (f"T: {T}\n"
                   f"R: {R}\n"
                   f"t: {t}\n"
                   f"delta: {delta}\n"
                   f"delta_inv: {delta_inv}\n\n")

        I_d = np.identity(d)
        zeros_d_d4 = np.zeros((d, d + 4))
        I_d4_d4_delta = delta_inv * np.identity(d + 4)

        number_of_combinations = 0
        success1 = 0
        success2 = 0
        successful_vectors = set()

        for w_d4_d in itertools.combinations(vectors, d + 4):

            number_of_combinations += 1
            M = np.block([
                [I_d, zeros_d_d4],
                [np.matrix(list(w_d4_d)), I_d4_d4_delta],
            ])
            M_LLL = olll.reduction(M.transpose().tolist(), 0.75)
            M_LLL_inv = np.matrix(M_LLL).transpose().tolist()

            for i in range(d):
                square = 1
                for j in range(d):
                    square *= pow(a_root[j], (M_LLL_inv[i][j]), N)
                    square %= N
                if (square * square) % N == 1:
                    break
            if (square * square) % N == 1:
                success1 += 1
                if square != N - 1 and square != 1:
                    print(f"Vector that gives p and q: {str(v)}")
                    p_q_vectors.append(str(v))
                    success2 += 1
                for v in w_d4_d:
                    successful_vectors.add(str(v))

        unsuccessful_vectors = vectors
        for v in successful_vectors:
            unsuccessful_vectors.remove(ast.literal_eval(v))

        end = time.time()
        exec_time = (end - start) * (10 ** 3)
        converted_time = convert_milliseconds(exec_time)

        result += (f"Number of combinations that result % N = 1: {success1 * 100 / number_of_combinations}%\n"
                   f"Number of combinations that give p and q: {success2 * 100 / number_of_combinations}%\n"
                   f"Unsuccessful vectors {unsuccessful_vectors}\n"
                   f"Successful vectors {successful_vectors}\n"
                   f"Vectors that gives p and q: {p_q_vectors}\n"
                   f"\nexec_time (ms): {exec_time} ms\n"
                   f"exec_time: {converted_time}")

        file = open(f"output_data/regev/classical_part/file_analysis/{dir1_part}_{dir2_part}/N_{N}", "w")
        file.write(result)
        file.close()

        print(f'Number of combinations that result % N = 1: {success1 * 100 / number_of_combinations}%')
        print(f'Number of combinations that give p and q: {success2 * 100 / number_of_combinations}%')
        print(f'Successful vectors {successful_vectors}')
        print(f'Vectors that give p and q: {p_q_vectors}')
        print(f'Unsuccessful vectors {unsuccessful_vectors}')

        print(f"exec_time: {exec_time}ms")
        print(f"converted_time: {converted_time}")


    def get_vectors(self, N: int, d_ceil=False, qd_ceil=False, semi_classical=False) -> 'RegevResult':
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
            vector = convert_measurement(measurement)
            self.result.output_data.append([vector, measurement, shots])
            self.vectors.append(vector)

            self.result.successful_counts += 1
            self.result.successful_shots += shots

        result = self.result
        self.result = RegevResult()

        return result


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


    def get_factors(self, vect, t_a, t_N):
        a = self.result.squared_primes
        N = self.result.N
        # a = t_a
        # N = t_N
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

        def run_on_quantum_computer(self, N: int, d_ceil=False, qd_ceil=False, semi_classical=False):
            self._validate_input(N)
    
            QiskitRuntimeService.save_account(channel="ibm_quantum", overwrite=True, token="API_TOKEN")
            service = QiskitRuntimeService()
            backend = service.least_busy(operational=True, simulator=False, min_num_qubits=127)
            circuit = self.construct_circuit(N, d_ceil, qd_ceil, semi_classical, measurement=True)
            print(circuit)
            print(f"Number of qubits: {circuit.num_qubits}")
            print(f"Number of classical bits: {circuit.num_clbits}")
            print(f'backbaend name: {backend.name}')
            pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
            isa_circuit = pm.run(circuit)
            print(isa_circuit)
            sampler = Sampler(backend)
            job = sampler.run([isa_circuit])
            result = job.result()
            print(f" > Counts: {result[0].data.meas.get_counts()}")


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



