import os
import ast
import math
import olll
import itertools
import numpy as np


def analize_vector(file_name):
    np.set_printoptions(suppress=True)
    if not os.path.exists(file_name):
        print(f"File {file_name} doesn't exists")
        return -1

    vectors = []

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

        while (line := results.readline()) != '\n':
            v = line.split(':')[1][:-2]
            duplicate = int(line.split(' ')[2])
            for i in range(min(d+4, duplicate)):
                vectors.append(ast.literal_eval(v))

        print(vectors)
        print(dq)

        T = 2
        R = math.ceil(6*T*math.sqrt((d+5)*(2*d)+4)*(d/2)*(2**((dq+1)/(d+4)+d+2)))
        t = 1 + math.ceil(math.log(math.sqrt(d)*R, 2))
        delta = math.sqrt(d/2)/R
        delta_inv = R/math.sqrt(d/2)

        I_d = np.identity(d)
        zeros_d_d4 = np.zeros((d, d + 4))
        I_d4_d4_delta = delta_inv * np.identity(d + 4)

        for w_d4_d in itertools.combinations(vectors, d+4):

            M = np.block([
                [I_d, zeros_d_d4],
                [np.matrix(list(w_d4_d)), I_d4_d4_delta],
            ])
            M_LLL= olll.reduction(M.transpose().tolist(), 0.75)
            M_LLL_inv = np.matrix(M_LLL).transpose().tolist()

            for i in range(d):
                square = 0
                for j in range(d):
                    square += a[j] ** (M_LLL_inv[i][j])
                if square % N == 1:
                    print(w_d4_d)


# A = np.identity(3)
# olll.reduction(A, 0.75)
analize_vector("./ceil_ceil/N_21")


