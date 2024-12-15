# This fragment of code allows to find exact value of T
import itertools
import math

for N, a_root in zip([15, 21, 33, 35, 39, 51, 55, 57], [[2,7], [2, 5, 11], [2, 5, 7], [2, 3, 11], [2, 5, 7], [2, 5, 7],
                                                    [2, 3, 7], [2, 5, 7]]):
    T = N
    n = N.bit_length()
    d = math.ceil(math.sqrt(n))
    m = math.ceil(n / d) + 2
    powers = []
    for i in range(m):
        powers.append(i)
    for p in itertools.product(powers, repeat=d):
        if p == (0,) * d:
            # print("UWAGA:", p)
            continue
        T_tmp = 1
        v_len_tmp = 0
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
    T_prim = math.exp(n/(2*d))
    print(T, T_prim)