import os
import ast
import math
import olll
import itertools
import numpy as np
from random import shuffle


def analize_vector(file_name, number_of_combinations):
    np.set_printoptions(suppress=True)
    if not os.path.exists(file_name):
        print(f"File {file_name} doesn't exists")
        return -1

    vectors = []

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

        # read vectors from file
        while (line := results.readline()) != '\n':
            v = line.split(':')[1][:-2]
            duplicate = int(line.split(' ')[2])
            for i in range(min(d+4, duplicate)):
                vectors.append(ast.literal_eval(v))

        # calculate parameters necessary to create lattice
        m = math.ceil(n/d) + 2
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
        R = math.ceil(6*T*math.sqrt((d+5)*(2*d)+4)*(d/2)*(2**((dq+1)/(d+4)+d+2)))
        t = 1 + math.ceil(math.log(math.sqrt(d)*R, 2))
        delta = math.sqrt(d/2)/R
        delta_inv = math.ceil(R/math.sqrt(d/2))
        print(f"Parameters:\nN: {N}\nR: {R}\nT: {T}\nt: {t}\ndelta: {delta}\ndelta_inv {delta_inv}")

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
            w_d4_d = vectors[:d+4]
            # create lattice M with usage created blocks according to Regev algorithm
            M = np.block([
                [I_d, zeros_d_d4],
                [np.matrix(w_d4_d)*(delta_inv/(2**t)), I_d4_d4_delta],
            ])

            # make LLL algorithm on columns of lattice M
            M_LLL= olll.reduction(M.transpose().tolist(), 0.75)
            M_LLL_t = np.matrix(M_LLL).transpose().tolist()

            # create flags to count different solutions from lattice once
            s1 = 0
            s2 = 0
            # s1_f = 0
            # s2_f = 0
            # check if given combinations of vectors returns correct solution

            for i in range(d):
                square = 1
                f = 0
                for j in range(d):
                    square *= pow(a_root[j], (M_LLL_t[i][j]), N)
                    square %= N
                    # if M_LLL_t[i][j] < 0:
                    #     f = 1
                if (square*square) % N == 1 and f == 0:
                    s1 = 1
                    if square != N - 1 and square != 1:
                        s2 = 1
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


        print(f'Per cent of combinations (with positive values of result vector) that gives % N = 1: {success1*100/number_of_combinations}%')
        print(f'Per cent of combinations (with positive values of result vector) that give p and q: {success2*100/number_of_combinations}%')

        # print(
        #     f'Per cent of combinations (including negative values) that gives % N = 1: {(success1_f + success1) * 100 / number_of_combinations}%')
        # print(
        #     f'Per cent of combinations (including negative values) that give p and q: {(success2_f + success2)* 100 / number_of_combinations}%')
        # print(f'Successful vectors {successful_vectors}')
        # unsuccessful_vectors = vectors
        # for v in successful_vectors:
        #     unsuccessful_vectors.remove(ast.literal_eval(v))
        # print(f'Unsuccessful vectors {unsuccessful_vectors}')




# A = np.identity(3)
# olll.reduction(A, 0.75)
#[15, 21, 33, 35, 39, 51, 55, 57]
for number in [15, 21, 33, 35, 39, 51, 55, 57]:
    analize_vector(f"./quantum_part/ceil_ceil/N_{number}", 10000)
