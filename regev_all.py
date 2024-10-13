
from implementations.r_haner import HanerRegev as Regev
from utils.convert_to_matrix_row import convert_to_matrix_row
from utils.convert_milliseconds import convert_milliseconds
import time

Ns = [15, 21, 33, 35, 39, 55, 65, 77, 143]
# Ns = [15, 21, 33]

d_ceil_bool = True
qd_ceil_bool = False

for i in range(len(Ns)):
    start = time.time()
    regev = Regev(shots=128)
    N=Ns[i]
    result = regev.get_vector(N, d_ceil=d_ceil_bool, qd_ceil=qd_ceil_bool, semi_classical=False)
    end = time.time()
    exec_time = (end-start)*(10**3)

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


    file = open(f"output_data/{d_mode}_{qd_mode}/N_{N}", "w")
    file.write(result_str)
    file.close()

    print("\n\n")
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


