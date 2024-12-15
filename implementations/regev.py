from typing import Union, Tuple, Optional

import numpy as np
from abc import ABC, abstractmethod
from itertools import chain, combinations

from qiskit import QuantumRegister, AncillaRegister, QuantumCircuit, ClassicalRegister, transpile

from qiskit.circuit import Instruction
from qiskit.circuit.library import QFT
from qiskit.visualization.circuit import matplotlib

from utils.circuit_creation import create_circuit
from utils.is_prime import is_prime
from utils.convert_measurement import convert_measurement
from utils.convert_to_matrix_row import convert_to_matrix_row
from utils.convert_milliseconds import convert_milliseconds

import logging
import math
import olll
from random import shuffle, randint
from fractions import Fraction
from decimal import Decimal, getcontext
import time
# from utils.secrets import ibm_api_token


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
from qiskit_ibm_runtime import SamplerV2 as Sampler

#from qiskit.utils.validation import validate_min

logger = logging.getLogger(__name__)
getcontext().prec = 1000


class Regev(ABC):

    def __init__(self,  shots) -> None:
        self.shots = shots
        self.result = RegevResult()
        self.vectors = []


    def draw_quantum_circuit(self, Ns, d_qd_list, decompose=False):

        for i in range(len(d_qd_list)):
            d_ceil_bool = d_qd_list[i][0]
            qd_ceil_bool = d_qd_list[i][1]

            for j in range(len(Ns)):

                N = Ns[j]

                if d_ceil_bool:
                    d_mode = "ceil"
                else:
                    d_mode = "floor"

                if qd_ceil_bool:
                    qd_mode = "ceil"
                else:
                    qd_mode = "floor"

                circuit = self.construct_circuit(N, d_ceil_bool, qd_ceil_bool)

                if decompose:
                    circuit.decompose().draw(output='mpl', filename=f'images/decomposed/{d_mode}_{qd_mode}/N_{N}.png', style='iqp-dark', fold=-1)
                else:
                    circuit.draw(output='mpl', filename=f'images/general/{d_mode}_{qd_mode}/N_{N}.png', style='iqp-dark', fold=-1)


    def run_all_algorithm(self, Ns, d_qd_list, number_of_combinations, type_of_test, find_pq=False):
        for i in range(len(d_qd_list)):
            d_ceil_bool = d_qd_list[i][0]
            qd_ceil_bool = d_qd_list[i][1]

            for j in range(len(Ns)):

                result_str = ""
                N = Ns[j]

                if d_ceil_bool:
                    d_mode = "ceil"
                else:
                    d_mode = "floor"

                if qd_ceil_bool:
                    qd_mode = "ceil"
                else:
                    qd_mode = "floor"

                print(f"\nN: {N}")
                start = time.time()
                quantum_result = self.get_vectors(N, d_ceil=d_ceil_bool, qd_ceil=qd_ceil_bool, semi_classical=False)
                classic_result = self.run_classical_part(number_of_combinations, N, quantum_result.n, quantum_result.number_of_primes, quantum_result.exp_register_width, quantum_result.squared_primes, quantum_result.output_data, type_of_test, find_pq)
                end = time.time()
                exec_time = (end - start) * 1000
                converted_time = convert_milliseconds(exec_time)

                result_str += (f"=============== QUANTUM PART ===============\n"
                               f"N: {quantum_result.N}\n"
                               f"n: {quantum_result.n}\n"
                               f"d_ceil: {quantum_result.d_ceil}\n"
                               f"qd_ceil: {quantum_result.qd_ceil}\n"
                               f"number_of_primes (d): {quantum_result.number_of_primes}\n"
                               f"exp_register_width (qd): {quantum_result.exp_register_width}\n"
                               f"squared_primes: {quantum_result.squared_primes}\n"
                               f"output_data: {quantum_result.output_data}\n"
                               f"\nvectors: {quantum_result}\n"
                               f"\nquantum part exec_time (ms): {quantum_result.quantum_exec_time}ms\n"
                               f"quantum part exec_time: {convert_milliseconds(quantum_result.quantum_exec_time)}\n")

                result_str += (f"\n=============== CLASSICAL PART ===============\n"
                               f"R: {classic_result.R}\n"
                               f"T: {classic_result.T}\n"
                               f"t: {classic_result.t}\n"
                               f"delta: {classic_result.delta}\n"
                               f"delta_inv: {classic_result.delta_inv}\n"
                               f"type_of_test: {type_of_test}\n"
                               f"number_of_combinations: {number_of_combinations}\n"
                               f"\nclassical part exec_time (ms): {classic_result.classical_exec_time}ms\n"
                               f"classical part exec_time: {convert_milliseconds(classic_result.classical_exec_time)}\n")


                result_str += f"\n=============== ALL TOGETHER ===============\n"

                if find_pq:
                    result_str += (f"p: {classic_result.p}\n"
                                   f"q: {classic_result.q}\n")

                result_str += (f"total exec_time (ms): {exec_time}ms\n"
                              f"total exec_time: {converted_time}")

                file = open(f"output_data/regev/all_parts/{d_mode}_{qd_mode}/N_{N}", "w")
                file.write(result_str)
                file.close()

        return 0


    def run_classical_part(self, number_of_combinations, N, n, d, qd, a, output_data, type_of_test, find_pq=False):

        print("Running classical part")
        # start = time.time()
        classic_result = RegevResult()
        vectors = []
        p_q_vectors = []
        a_root = []

        start = time.time()

        for a_ in a:
            a_root.append(int(math.sqrt(a_)))

        # output_data[i]: [Vector, measurments, shots]
        total_number_of_vectors = 0
        for i in range(len(output_data)):
            duplicate = output_data[i][2]
            if type_of_test == 1:
                for j in range(min(d + 4, duplicate)):
                    vectors.append(output_data[i][0])
            if type_of_test == 2:
                vectors.append(output_data[i][0])
            if type_of_test == 3:
                total_number_of_vectors += duplicate

        if type_of_test == 3:
            for i in range(total_number_of_vectors):
                v = []
                for j in range(d):
                    v.append(randint(0, 2 ** qd))
                vectors.append(v)
        if type_of_test == 2 and len(vectors) < d + 4:
            print(f"\nToo little variety of vectors for number {N}\n")
            return -1

        # calculate parameters necessary to create lattice
        m = math.ceil(n / d) + 2
        powers = []
        for i in range(m):
            powers.append(i)

        T = N

        for p in itertools.product(powers, repeat=d):
            if p == (0,) * d:
                # print("UWAGA:", p)
                continue
            T_tmp = 1
            v_len_tmp = 1
            for i in range(d):
                T_tmp *= pow(a_root[i], p[i], N)
                v_len_tmp += pow(p[i], 2)
            v_len_tmp = math.ceil(math.sqrt(v_len_tmp))
            # print(p, T_tmp, v_len_tmp)
            if T_tmp % N == 1 and v_len_tmp < T:
                # print(a_root)
                # print(p)
                # print(v_len_tmp)
                T = v_len_tmp

        # print('T', T)
        R = math.ceil(6 * T * math.sqrt((d + 5) * (2 * d) + 4) * (d / 2) * (2 ** ((qd + 1) / (d + 4) + d + 2)))
        t = 1 + math.ceil(math.log(math.sqrt(d) * R, 2))
        delta = math.sqrt(d / 2) / R
        delta_inv = math.ceil(R / math.sqrt(d / 2))
        print(f"Parameters:\nN: {N}\nR: {R}\nT: {T}\nt: {t}\ndelta: {delta}\ndelta_inv: {delta_inv}")

        classic_result.R = R
        classic_result.T = T
        classic_result.t = t
        classic_result.delta = delta
        classic_result.delta_inv = delta_inv

        # create block of lattice
        I_d = np.identity(d)
        zeros_d_d4 = np.zeros((d, d + 4))
        I_d4_d4_delta = delta_inv * np.identity(d + 4)

        success1 = 0
        success2 = 0
        # success1_f = 0
        # success2_f = 0

        for _ in range(number_of_combinations):
            # get random combinations from vectors
            shuffle(vectors)
            w_d4_d = vectors[:d + 4]
            # create lattice M with usage created blocks according to Regev algorithm
            M = np.block([
                [I_d, zeros_d_d4],
                [np.matrix(w_d4_d) * (delta_inv / (2 ** t)), I_d4_d4_delta],
            ])

            # make LLL algorithm on columns of lattice M
            M_LLL = olll.reduction(M.transpose().tolist(), 0.75)
            M_LLL_t = np.matrix(M_LLL).transpose().tolist()

            # create flags to count different solutions from lattice once
            s1 = 0
            s2 = 0
            # s1_f = 0
            # s2_f = 0
            # check if given combinations of vectors returns correct solution

            break_flag = 0
            for i in range(1, 2*d + 4):
                square = 1
                f = 0
                temp_vector = []
                for j in range(d):
                    square *= pow(a_root[j], (M_LLL_t[i][j]), N)
                    square %= N
                    temp_vector.append(M_LLL_t[i][j])
                    # if M_LLL_t[i][j] < 0:
                    #     f = 1
                if (square * square) % N == 1 and f == 0:
                    s1 = 1
                    if square != N - 1 and square != 1:
                        s2 = 1
                        # TODO: wyjść na dobre z tych pętli tak, żeby od razu iść zwrócić wektor (lub też wyliczyć p i q jeżeli find_pq=True)
                        p_q_vectors.append(temp_vector)
                        break_flag = 1
                        break
                if break_flag == 1:
                    break
                # if (square*square) % N == 1 and f == 1:
                #     s1_f = 1
                #     if square != N-1 and square != 1:
                #         s2_f = 1

            if s1 == 1:
                success1 += 1
            # elif s1_f == 1:
            #     success1_f += 1

            if s2 == 1:
                success2 += 1
            # elif s2_f == 1:
            #     success2_f += 1

        end = time.time()
        exec_time = (end - start) * (10 ** 3)
        classic_result.classical_exec_time = exec_time

        classic_result.vector = p_q_vectors[0]

        if find_pq:
            vector = p_q_vectors[0]
            p, q = self.get_factors(vector, a_root, N)
            classic_result.p = p
            classic_result.q = q

        return classic_result


    def run_quantum_part_data_collection(self, Ns, d_qd_list):

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


    def run_file_data_analyzer_new(self, Ns, d_qd_list, number_of_combinations, type_of_test=1):

        # Type of test
        # 1 - deafult, check number_of_combinations random combinations of vectors return by quantum computer if returns
        # correct powers, with probability according to this returned by quantum computer
        # 2 - check number_of_combinations random combinations of vectors return by quantum computer if returns
        # correct powers, but do not count fact that some vectors are replicated
        # 3 - check number_of_combinations random combinations of totaly random vectors if returns
        # correct powers

        for k in range(len(d_qd_list)):
            d_ceil_bool = d_qd_list[k][0]
            qd_ceil_bool = d_qd_list[k][1]

            if d_ceil_bool:
                d_mode = "ceil"
            else:
                d_mode = "floor"

            if qd_ceil_bool:
                qd_mode = "ceil"
            else:
                qd_mode = "floor"

            for l in range(len(Ns)):

                N = Ns[l]
                print(f"\nN: {N}")
                file_name = f"./output_data/regev/quantum_part/{d_mode}_{qd_mode}/N_{N}"

                if not os.path.exists(file_name):
                    print(f"File {file_name} doesn't exists")
                    return -1


                result = ""
                vectors = []
                p_q_vectors = []

                dir1_part = file_name.split("/")[-2].split("_")[0]
                dir2_part = file_name.split("/")[-2].split("_")[1]

                print(f"dir1_part: {dir1_part}\ndir2_part: {dir2_part}")

                with open(file_name) as results:

                    # read parameters from input file
                    dq = 0
                    for i in range(10):
                        line = results.readline()
                        if i == 0:
                            N = int(line.split(' ')[1])
                        if i == 1:
                            n = int(line.split(' ')[1])
                        if i == 4:
                            d = int(line.split(':')[1][:-1])
                        if i == 5:
                            dq = int(line.split(':')[1][:-1])
                        if i == 6:
                            a = ast.literal_eval(line.split(':')[1])
                            a_root = []
                            for a_ in a:
                                a_root.append(int(math.sqrt(a_)))

                    # read vectors from file or generate vectors
                    total_number_of_vectors = 0
                    while (line := results.readline()) != '\n':
                        v = line.split(':')[1][:-2]
                        duplicate = int(line.split(' ')[2])
                        if type_of_test == 1:
                            for i in range(duplicate):
                                vectors.append(ast.literal_eval(v))
                                # print(f"ast.literal_eval(v): {ast.literal_eval(v)}")
                        if type_of_test == 2:
                            vectors.append(ast.literal_eval(v))
                        if type_of_test == 3:
                            total_number_of_vectors += duplicate
                    if type_of_test == 3:
                        for i in range(total_number_of_vectors):
                            v = []
                            for j in range(d):
                                v.append(randint(0, 2**dq))
                            vectors.append(v)
                    if type_of_test == 2 and len(vectors) < d+4:
                        result += f"\nToo little variety of vectors for number {N}\n"
                        print(f"\nToo little variety of vectors for number {N}\n")
                        continue


                    start = time.time()

                    # calculate parameters necessary to create lattice
                    m = math.ceil(n / d) + 2
                    powers = []
                    for i in range(m):
                        powers.append(i)


                    # This fragment of code allows to find exact value of T
                    # T = N
                    # for p in itertools.product(powers, repeat=d):
                    #     if p == (0,) * d:
                    #         # print("UWAGA:", p)
                    #         continue
                    #     T_tmp = 1
                    #     v_len_tmp = 1
                    #     for i in range(d):
                    #         T_tmp *= pow(a_root[i], p[i], N)
                    #         v_len_tmp += pow(p[i], 2)
                    #     v_len_tmp = math.ceil(math.sqrt(v_len_tmp))
                    #     # print(p, T_tmp, v_len_tmp)
                    #     if T_tmp % N == 1 and v_len_tmp < T:
                    #         # print(a_root)
                    #         # print(p)
                    #         # print(v_len_tmp)
                    #         T = v_len_tmp

                    # This fragment of code estimate the value of T
                    T = math.ceil(math.exp(n/(2*d)))

                    R = math.ceil(6 * T * math.sqrt((d + 5) * (2 * d) + 4) * (d / 2) * (2 ** ((dq + 1) / (d + 4) + d + 2)))
                    t = 1 + math.ceil(math.log(math.sqrt(d) * R, 2))
                    delta = math.sqrt(d / 2) / R
                    delta_inv = math.ceil(R / math.sqrt(d / 2))
                    print(f"Parameters:\nN: {N}\nR: {R}\nT: {T}\nt: {t}\ndelta: {delta}\ndelta_inv: {delta_inv}")

                    result += (f"N: {N}\n"
                               f"n: {n}\n"
                               f"number_of_primes (d): {d}\n"
                               f"exp_register_width (qd): {dq}\n"
                               f"primes: {a_root}\n\n"
                               f"R: {R}\n"
                               f"T: {T}\n"
                               f"t: {t}\n"
                               f"delta: {delta}\n"
                               f"delta_inv: {delta_inv}")

                    # create block of lattice
                    I_d = np.identity(d)
                    zeros_d_d4 = np.zeros((d, d + 4))
                    I_d4_d4_delta = delta_inv * np.identity(d + 4)

                    success1 = 0
                    success2 = 0
                    # success1_f = 0
                    # success2_f = 0

                    for _ in range(number_of_combinations):
                        # get random combinations from vectors
                        shuffle(vectors)
                        w_d4_d = vectors[:d + 4]
                        # create lattice M with usage created blocks according to Regev algorithm
                        M = np.block([
                            [I_d, zeros_d_d4],
                            [np.matrix(w_d4_d) * (delta_inv / (2 ** t)), I_d4_d4_delta],
                        ])

                        # make LLL algorithm on columns of lattice M
                        M_LLL = olll.reduction(M.transpose().tolist(), 0.75)
                        M_LLL_t = np.matrix(M_LLL).transpose().tolist()
                        # create flags to count different solutions from lattice once
                        s1 = 0
                        s2 = 0
                        # s1_f = 0
                        # s2_f = 0
                        # check if given combinations of vectors returns correct solution

                        for i in range(0, 2*d + 4):
                            square = 1
                            f = 0
                            temp_vector = []
                            for j in range(d):
                                square *= pow(a_root[j], (M_LLL_t[i][j]), N)
                                square %= N
                                temp_vector.append(M_LLL_t[i][j])
                                # if M_LLL_t[i][j] < 0:
                                #     f = 1
                            if (square * square) % N == 1 and f == 0 and temp_vector != d*[0]:
                                s1 = 1
                                if square != N - 1 and square != 1:
                                    s2 = 1
                                    p_q_vectors.append(temp_vector)
                                    break
                            # if (square*square) % N == 1 and f == 1:
                            #     s1_f = 1
                            #     if square != N-1 and square != 1:
                            #         s2_f = 1

                        if s1 == 1:
                            success1 += 1
                        # elif s1_f == 1:
                        #     success1_f += 1

                        if s2 == 1:
                            success2 += 1
                        # elif s2_f == 1:
                        #     success2_f += 1


                end = time.time()
                exec_time = (end - start) * (10 ** 3)
                converted_time = convert_milliseconds(exec_time)

                result += (f"Percent of combinations (with positive values of result vector) that gives % N = 1: {success1 * 100 / number_of_combinations}%\n"
                           f"Percent of combinations (with positive values of result vector) that give p and q: {success2 * 100 / number_of_combinations}%\n"
                           # f"Percent of combinations (including negative values) that gives % N = 1: {(success1_f + success1) * 100 / number_of_combinations}%\n"
                           # f"Percent of combinations (including negative values) that give p and q: {(success2_f + success2) * 100 / number_of_combinations}%\n"
                           # f"Unsuccessful vectors {unsuccessful_vectors}\n"
                           # f"Successful vectors {successful_vectors}\n"
                           f"Vectors that gives p and q: {p_q_vectors}\n"
                           f"\nexec_time (ms): {exec_time} ms\n"
                           f"exec_time: {converted_time}")


                file = open(f"output_data/regev/classical_part/file_analysis_2/{dir1_part}_{dir2_part}/N_{N}", "w")
                file.write(result)
                file.close()

                print(f'Per cent of combinations (with positive values of result vector) that gives % N = 1: {success1 * 100 / number_of_combinations}%')
                print(f'Per cent of combinations (with positive values of result vector) that give p and q: {success2 * 100 / number_of_combinations}%')
                # print(f'Percent of combinations (including negative values) that gives % N = 1: {(success1_f + success1) * 100 / number_of_combinations}%')
                # print(f'Percent of combinations (including negative values) that give p and q: {(success2_f + success2) * 100 / number_of_combinations}%')
                print(f"Vectors that gives p and q: {p_q_vectors}")
                print(f"\nexec_time: {exec_time} ms")
                print(f"exec_time: {converted_time}")

                # This code is temporary, needs to be deleted
                if len(p_q_vectors) > 0:
                    print(f"CALCULATING P AND Q")

                    vector = p_q_vectors[0]
                    self.get_factors(vector, a_root, N)


    def get_vectors(self, N: int, d_ceil=False, qd_ceil=False, semi_classical=False) -> 'RegevResult':

        print("Running quantum part")

        start = time.time()
        self._validate_input(N)

        circuit = self.construct_circuit(N, d_ceil, qd_ceil, semi_classical, measurement=True)
        # aersim = AerSimulator(method="extended_stabilizer")
        aersim = AerSimulator()
        # Wyświetlenie liczby obsługiwanych kubitów
        print("Max number of qubits (local qasm_simulator):", aersim.configuration().n_qubits)
        # print("Max circuits: ", aersim.max_circuits)
        # print("available_methods(): ", aersim.available_methods())
        # print("available_devices(): ", aersim.available_devices())
        #
        # print("configuration(): ", aersim.configuration())
        # print("name: ", aersim.name)
        # print("")


        # pm = generate_preset_pass_manager(backend=aersim, optimization_level=3  )
        pm = transpile(circuit, aersim)

        # isa_qc = pm.run(circuit)
        counts = aersim.run(pm, shots=self.shots).result().get_counts(0)

        # counts = aersim.run(isa_qc, shots=self.shots).result().get_counts(0)

        # counts = aersim.run(isa_qc, shots=self.shots).result().get_counts(0)
        # counts = result.get_counts(0)
        # print('Counts(ideal):', counts)

        # counts=self.sampler().run(circuit, shots=self.shots).result().quasi_dists[0].binary_probabilities()

        self.result.total_counts = len(counts)
        self.result.total_shots = self.shots
        # print(f"counts.items(): {counts.items()}")

        sorted_counts_items = sorted(counts.items(), key=lambda x: x[1])

        for measurement, shots in sorted_counts_items:
            # measurement = self._parse_measurement(measurement, semi_classical)
            # print(f", measurment: {measurement}   |   shots: {shots}", end="")
            vector = convert_measurement(measurement)
            self.result.output_data.append([vector, measurement, shots])

            # The following two lines might be useless
            self.vectors.append(vector)
            self.result.vectors.append(vector)

            self.result.successful_counts += 1
            self.result.successful_shots += shots

        end = time.time()
        exec_time = (end - start) * (10 ** 3)
        self.result.quantum_exec_time = exec_time
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


    @staticmethod
    def get_factors(vect, primes, N):

        print(f"squared_primes = {primes}\n"
              f"vect: {vect}\n")

        # prod = Decimal(1)
        prod = 1

        for i in range(len(primes)):
            # sqrt_a = Decimal(primes[i]).sqrt()
            # pow_a = ((Decimal(primes[i]) ** vect[i]) % N)
            # pow_a = pow(Decimal(primes[i]), vect[i], N)
            # prod = ((prod*pow_a) % N)
            prod *= pow(primes[i], (vect[i]), N)
            prod %= N

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

        print(f"p: {p}\nq: {q}")

        return p, q


    def run_on_quantum_computer(self, N: int, d_ceil=False, qd_ceil=False, semi_classical=False):
        self._validate_input(N)
        result_str = ""

        # QiskitRuntimeService.save_account(channel="ibm_quantum", overwrite=True, token=ibm_api_token)
        service = QiskitRuntimeService()
        # backend = service.least_busy(operational=True, simulator=False, min_num_qubits=127)
        backend = service.backend("ibm_sherbrooke")
        circuit = self.construct_circuit(N, d_ceil, qd_ceil, semi_classical, measurement=True)
        print(circuit)
        print(f"Number of qubits: {circuit.num_qubits}")

        print(f"Number of classical bits: {circuit.num_clbits}")
        print(f'Backend name: {backend.name}')
        pm = generate_preset_pass_manager(backend=backend, optimization_level=0)
        isa_circuit = pm.run(circuit)
        print(isa_circuit)
        sampler = Sampler(backend)
        job = sampler.run([isa_circuit])
        result = job.result()
        print(f" > Counts: {result[0].data.meas.get_counts()}")

        if d_ceil:
            d_mode = "ceil"
        else:
            d_mode = "floor"

        if qd_ceil:
            qd_mode = "ceil"
        else:
            qd_mode = "floor"

        result_str += (f"Number of qubits: {circuit.num_qubits}\n"
                       f"Number of classical bits: {circuit.num_clbits}\n"
                       f"Backend name: {backend.name}\n"
                       f" > Counts: {result[0].data.meas.get_counts()}")

        file = open(f"output_data/regev/quantum_computer/{d_mode}_{qd_mode}/N_{N}", "w")
        file.write(result_str)
        file.close()


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
        self._vectors = []
        self._quantum_exec_time = 0

        self._R = 0
        self._T = 0
        self._t = 0
        self._delta = 0
        self._delta_inv = 0
        self._vector = 0
        self._p = 0
        self._q = 0
        self._classical_exec_time = 0


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

    @property
    def vectors(self) -> []:
        return self._vectors

    @vectors.setter
    def vectors(self, value: []) -> None:
        self._vectors = value

    @property
    def quantum_exec_time(self) -> int:
        return self._quantum_exec_time

    @quantum_exec_time.setter
    def quantum_exec_time(self, value: int) -> None:
        self._quantum_exec_time = value



    @property
    def R(self) -> int:
        return self._R

    @R.setter
    def R(self, value: int) -> None:
        self._R = value

    @property
    def T(self) -> int:
        return self._T

    @T.setter
    def T(self, value: int) -> None:
        self._T = value

    @property
    def t(self) -> int:
        return self._t

    @t.setter
    def t(self, value: int) -> None:
        self._t = value

    @property
    def delta(self) -> int:
        return self._delta

    @delta.setter
    def delta(self, value: int) -> None:
        self._delta = value

    @property
    def delta_inv(self) -> int:
        return self._delta_inv

    @delta_inv.setter
    def delta_inv(self, value: int) -> None:
        self._delta_inv = value

    @property
    def vector(self) -> []:
        return self._vector

    @vector.setter
    def vector(self, value: []) -> None:
        self._vector = value

    @property
    def p(self) -> int:
        return self._p

    @p.setter
    def p(self, value: int) -> None:
        self._p = value

    @property
    def q(self) -> int:
        return self._q

    @q.setter
    def q(self, value: int) -> None:
        self._q = value

    @property
    def classical_exec_time(self) -> int:
        return self._classical_exec_time

    @classical_exec_time.setter
    def classical_exec_time(self, value: int) -> None:
        self._classical_exec_time = value


